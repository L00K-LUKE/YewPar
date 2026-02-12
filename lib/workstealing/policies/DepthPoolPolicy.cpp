#include "DepthPoolPolicy.hpp"

#include <hpx/functional/function.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

#include <algorithm>
#include <memory>
#include <random>

#include "util/util.hpp"

namespace Workstealing { namespace Policies {

namespace DepthPoolPolicyPerf {

std::atomic<std::uint64_t> perf_spawns(0);
std::atomic<std::uint64_t> perf_localSteals(0);
std::atomic<std::uint64_t> perf_distributedSteals(0);
std::atomic<std::uint64_t> perf_failedLocalSteals(0);
std::atomic<std::uint64_t> perf_failedDistributedSteals(0);

std::uint64_t get_and_reset(std::atomic<std::uint64_t> & cntr, bool reset) {
  auto res = cntr.load();
  if (reset) { cntr = 0; }
  return res;
}

std::uint64_t getSpawns (bool reset) { return get_and_reset(perf_spawns, reset);}
std::uint64_t getLocalSteals(bool reset) { return get_and_reset(perf_localSteals, reset);}
std::uint64_t getDistributedSteals (bool reset) { return get_and_reset(perf_distributedSteals, reset);}
std::uint64_t getFailedLocalSteals(bool reset) { return get_and_reset(perf_failedLocalSteals, reset);}
std::uint64_t getFailedDistributedSteals(bool reset) { return get_and_reset(perf_failedDistributedSteals, reset);}

void registerPerformanceCounters() {
  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/spawns",
      &getSpawns,
      "Returns the number of tasks spawned on this locality"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/localSteals",
      &getLocalSteals,
      "Returns the number of tasks stolen from another thread on the same locality"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/distributedSteals",
      &getDistributedSteals,
      "Returns the number of tasks stolen from another thread on another locality"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/localFailedSteals",
      &getFailedLocalSteals,
      "Returns the number of failed steals from this locality "
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/distributedFailedSteals",
      &getFailedDistributedSteals,
      "Returns the number of failed steals from another locality "
                                                  );
}

}

DepthPoolPolicy::DepthPoolPolicy(hpx::id_type workpool) {
  local_workpool = workpool;
  last_remote = hpx::find_here();
}

void DepthPoolPolicy::rebuildVictimOrder() {
  victim_order = distributed_workpools;
  std::mt19937 deterministic_rng(
      static_cast<std::mt19937::result_type>(hpx::get_locality_id()));
  std::shuffle(victim_order.begin(), victim_order.end(), deterministic_rng);
  next_victim_idx = 0;
}

void DepthPoolPolicy::resumeAfterVictim(hpx::id_type victim) {
  auto it = std::find(victim_order.begin(), victim_order.end(), victim);
  if (it == victim_order.end() || victim_order.empty()) {
    return;
  }

  auto idx = static_cast<std::size_t>(std::distance(victim_order.begin(), it));
  next_victim_idx = (idx + 1) % victim_order.size();
}

hpx::function<void(), false> DepthPoolPolicy::getWork() {
  std::unique_lock<mutex_t> l(mtx);

  hpx::distributed::function<void(hpx::id_type)> task;
  task = hpx::async<workstealing::DepthPool::getLocal_action>(local_workpool).get();

  if (task) {
    DepthPoolPolicyPerf::perf_localSteals++;
    return hpx::bind(task, hpx::find_here());
  } else {
    DepthPoolPolicyPerf::perf_failedLocalSteals++;
  }

  if (!victim_order.empty()) {
    hpx::id_type attempted_victim = hpx::find_here();
    bool had_attempted_victim = false;

    // Last steal optimisation
    if (last_remote != hpx::find_here()) {
      attempted_victim = last_remote;
      had_attempted_victim = true;
      task = hpx::async<workstealing::DepthPool::steal_action>(last_remote).get();
      if (task) {
        resumeAfterVictim(last_remote);
        DepthPoolPolicyPerf::perf_distributedSteals++;
        return hpx::bind(task, hpx::find_here());
      } else {
        resumeAfterVictim(last_remote);
        DepthPoolPolicyPerf::perf_failedDistributedSteals++;
        last_remote = hpx::find_here();
      }
    }

    auto victim = victim_order[next_victim_idx];
    if (!had_attempted_victim || victim != attempted_victim) {
      task = hpx::async<workstealing::DepthPool::steal_action>(victim).get();
      resumeAfterVictim(victim);

      if (task) {
        last_remote = victim;
        DepthPoolPolicyPerf::perf_distributedSteals++;
        return hpx::bind(task, hpx::find_here());
      } else {
        DepthPoolPolicyPerf::perf_failedDistributedSteals++;
      }
    }
  }

  return nullptr;
}

void DepthPoolPolicy::addwork(hpx::distributed::function<void(hpx::id_type)> task, unsigned depth) {
  std::unique_lock<mutex_t> l(mtx);
  DepthPoolPolicyPerf::perf_spawns++;
  hpx::post<workstealing::DepthPool::addWork_action>(local_workpool, task, depth);
}

void DepthPoolPolicy::registerDistributedDepthPools(std::vector<hpx::id_type> workpools) {
  std::unique_lock<mutex_t> l(mtx);
  distributed_workpools = workpools;
  distributed_workpools .erase(
      std::remove_if(distributed_workpools.begin(), distributed_workpools.end(), YewPar::util::isColocated),
      distributed_workpools.end());
  rebuildVictimOrder();
  last_remote = hpx::find_here();
}

}}
