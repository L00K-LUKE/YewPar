#include "DepthPoolPolicy.hpp"

#include <hpx/functional/function.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

#include <algorithm>
#include <memory>

#include "util/util.hpp"

namespace Workstealing { namespace Policies {

std::atomic<bool> DepthPoolPolicy::use_last_steal{true};

namespace DepthPoolPolicyPerf {

std::atomic<std::uint64_t> perf_spawns(0);
std::atomic<std::uint64_t> perf_localSteals(0);
std::atomic<std::uint64_t> perf_distributedSteals(0);
std::atomic<std::uint64_t> perf_failedLocalSteals(0);
std::atomic<std::uint64_t> perf_failedDistributedSteals(0);
std::atomic<std::uint64_t> perf_lastStealSuccesses(0);

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
std::uint64_t getLastStealSuccesses(bool reset) { return get_and_reset(perf_lastStealSuccesses, reset);}

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

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/lastStealSuccesses",
      &getLastStealSuccesses,
      "Returns the number of successful steals via the last steal optimisation"
                                                  );
}

}

DepthPoolPolicy::DepthPoolPolicy(hpx::id_type workpool) {
  local_workpool = workpool;
  last_remote = hpx::find_here();
  randGenerator.seed(static_cast<std::mt19937::result_type>(hpx::get_locality_id()));
}

hpx::function<void(), false> DepthPoolPolicy::getWork() {
  hpx::distributed::function<void(hpx::id_type)> task;
  task = hpx::async<workstealing::DepthPool::getLocal_action>(local_workpool).get();

  auto here = hpx::find_here();

  if (task) {
    DepthPoolPolicyPerf::perf_localSteals++;
    return hpx::bind(task, here);
  } else {
    DepthPoolPolicyPerf::perf_failedLocalSteals++;
  }

  hpx::id_type preferred_victim = here;
  std::vector<hpx::id_type> victim_order;
  std::size_t victim_start_index = 0;

  {
    std::unique_lock<mutex_t> l(mtx);
    if (!distributed_workpools.empty()) {
      preferred_victim = last_remote;
      victim_order = distributed_workpools;
      victim_start_index = next_victim_index % victim_order.size();
    }
  }

  if (!victim_order.empty()) {
    bool attempted_last_victim = false;

    // Last steal optimisation
    if (use_last_steal.load(std::memory_order_relaxed) && preferred_victim != here) {
      attempted_last_victim = true;
      task = hpx::async<workstealing::DepthPool::steal_action>(preferred_victim).get();
      if (task) {
        std::unique_lock<mutex_t> l(mtx);
        last_remote = preferred_victim;
        auto victim_it = std::find(victim_order.begin(), victim_order.end(), preferred_victim);
        if (victim_it != victim_order.end()) {
          auto victim_index = static_cast<std::size_t>(std::distance(victim_order.begin(), victim_it));
          next_victim_index = (victim_index + 1) % victim_order.size();
        }
        DepthPoolPolicyPerf::perf_lastStealSuccesses++;
        DepthPoolPolicyPerf::perf_distributedSteals++;
        return hpx::bind(task, here);
      }

      DepthPoolPolicyPerf::perf_failedDistributedSteals++;
      {
        std::unique_lock<mutex_t> l(mtx);
        if (last_remote == preferred_victim) {
          last_remote = here;
        }
      }
    }

    // Deterministic steal order for the remaining victims.
    for (std::size_t i = 0; i < victim_order.size(); ++i) {
      auto victim_index = (victim_start_index + i) % victim_order.size();
      auto const& victim = victim_order[victim_index];

      if (victim == here || (attempted_last_victim && victim == preferred_victim)) {
        continue;
      }

      task = hpx::async<workstealing::DepthPool::steal_action>(victim).get();
      if (task) {
        std::unique_lock<mutex_t> l(mtx);
        last_remote = victim;
        next_victim_index = (victim_index + 1) % victim_order.size();
        DepthPoolPolicyPerf::perf_distributedSteals++;
        return hpx::bind(task, here);
      }

      DepthPoolPolicyPerf::perf_failedDistributedSteals++;
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
  distributed_workpools.erase(
      std::remove_if(distributed_workpools.begin(), distributed_workpools.end(), YewPar::util::isColocated),
      distributed_workpools.end());
  std::shuffle(distributed_workpools.begin(), distributed_workpools.end(), randGenerator);
  next_victim_index = 0;
}

}}
