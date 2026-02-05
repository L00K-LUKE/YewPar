#include "DepthPoolPolicy.hpp"

#include <hpx/functional/function.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

#include <memory>

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
  local_workpool_jobs.store(
      hpx::async<workstealing::DepthPool::size_action>(local_workpool).get());

  std::random_device rd;
  randGenerator.seed(rd());
}

std::vector<std::size_t> DepthPoolPolicy::getIndices() {
  std::vector<std::size_t> indices(distributed_workpools.size());
  std::iota(indices.begin(), indices.end(), 0);

  for (std::size_t i = 0; i < distributed_sample_size; ++i) {
    std::uniform_int_distribution<size_t> dist(i, indices.size() - 1);
    size_t j = dist(randGenerator);
    std::swap(indices[i], indices[j]);
  }
  indices.resize(distributed_sample_size);
  return indices;
}

hpx::id_type DepthPoolPolicy::getBestVictim(const std::vector<std::size_t>& indices) {

  std::size_t biggest_pool = 0;
  hpx::id_type best_victim = hpx::invalid_id;

  for (std::size_t i : indices) {
    auto candidate = distributed_workpools[i];
    auto candidate_size = hpx::async<workstealing::DepthPool::size_action>(candidate).get();
    if (candidate_size > biggest_pool) {
      biggest_pool = candidate_size;
      best_victim = candidate;
    }
  }
  return best_victim;
}

hpx::function<void(), false> DepthPoolPolicy::getWork() {
  std::unique_lock<mutex_t> l(mtx);

  hpx::distributed::function<void(hpx::id_type)> task;
  task = hpx::async<workstealing::DepthPool::getLocal_action>(local_workpool).get();

  // Steal from local pool
  if (task) { 
    DepthPoolPolicyPerf::perf_localSteals++;
    auto current_jobs = local_workpool_jobs.load();
    if (current_jobs > 0) {
      local_workpool_jobs.fetch_sub(1);
    }
    return hpx::bind(task, hpx::find_here());
  } else {
    DepthPoolPolicyPerf::perf_failedLocalSteals++;
  }

  // Steal from last successful remote steal
  if (last_remote != hpx::find_here()) {
    task = hpx::async<workstealing::DepthPool::steal_action>(last_remote).get();
    if (task) {
      DepthPoolPolicyPerf::perf_distributedSteals++;
      return hpx::bind(task, hpx::find_here());
    } else {
      DepthPoolPolicyPerf::perf_failedDistributedSteals++;
    }
  }

  // Try to steal from a random sample of the distributed pools. Pick the one with the most work to steal from.
  if (!distributed_workpools.empty()) {
    const std::vector<std::size_t> indices = getIndices();
    const auto best_victim = getBestVictim(indices);
    
    if (best_victim == hpx::invalid_id) {
      std::cout << "No valid victim found for stealing" << std::endl;
      return nullptr;
    }

    task = hpx::async<workstealing::DepthPool::steal_action>(best_victim).get();
    if (task) {
      last_remote = best_victim;
      DepthPoolPolicyPerf::perf_distributedSteals++;
      return hpx::bind(task, hpx::find_here());
    } else {
      DepthPoolPolicyPerf::perf_failedDistributedSteals++;
    }
    // TODO: Could try stealing in order of most work to least work.
    // Also maybe worth trying random steal?
    // For now, just return nullptr if the steal fails.
    return nullptr;
  }
  return nullptr; 
}

void DepthPoolPolicy::addwork(hpx::distributed::function<void(hpx::id_type)> task, unsigned depth) {
  std::unique_lock<mutex_t> l(mtx);
  DepthPoolPolicyPerf::perf_spawns++;
  local_workpool_jobs.fetch_add(1);
  hpx::post<workstealing::DepthPool::addWork_action>(local_workpool, task, depth);
}

void DepthPoolPolicy::registerDistributedDepthPools(std::vector<hpx::id_type> workpools) {
  std::unique_lock<mutex_t> l(mtx);
  distributed_workpools = workpools;
  distributed_workpools .erase(
      std::remove_if(distributed_workpools.begin(), distributed_workpools.end(), YewPar::util::isColocated),
      distributed_workpools.end());
}

}}
