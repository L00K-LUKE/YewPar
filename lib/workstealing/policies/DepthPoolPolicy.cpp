#include "DepthPoolPolicy.hpp"

#include <hpx/functional/function.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

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
std::uint64_t getlastStealSuccesses(bool reset) { return get_and_reset(perf_lastStealSuccesses, reset);}

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
      &getlastStealSuccesses,
      "Returns the number of successful steals from the previous remote victim"
                                                  );
}

}

DepthPoolPolicy::DepthPoolPolicy(hpx::id_type workpool) {
  local_workpool = workpool;
  last_remote = hpx::find_here();

  std::random_device rd;
  randGenerator.seed(rd());
}

DepthPoolPolicy::task_t DepthPoolPolicy::getLocalTask() {
  return hpx::async<workstealing::DepthPool::getLocal_action>(local_workpool).get();
}

bool DepthPoolPolicy::tryGetLastRemoteVictim(hpx::id_type here, hpx::id_type & victim) {
  if (!use_last_steal.load(std::memory_order_relaxed)) {
    return false;
  }

  std::unique_lock<mutex_t> l(mtx);
  if (last_remote == here) {
    return false;
  }

  victim = last_remote;
  return true;
}

std::vector<hpx::id_type> DepthPoolPolicy::chooseDistributedCandidates() {
  std::vector<hpx::id_type> candidates;
  candidates.reserve(2);

  std::unique_lock<mutex_t> l(mtx);
  if (distributed_workpools.empty()) {
    return candidates;
  }

  std::uniform_int_distribution<std::size_t> rand(0, distributed_workpools.size() - 1);

  auto first_idx = rand(randGenerator);
  candidates.push_back(distributed_workpools[first_idx]);

  if (distributed_workpools.size() > 1) {
    auto second_idx = first_idx;
    while (second_idx == first_idx) {
      second_idx = rand(randGenerator);
    }
    candidates.push_back(distributed_workpools[second_idx]);
  }

  return candidates;
}

hpx::id_type DepthPoolPolicy::chooseBestVictim(const std::vector<hpx::id_type> & candidates) {

  std::vector<hpx::future<std::size_t>> poll_results;
  poll_results.reserve(candidates.size());

  for (auto const& victim : candidates) {
    poll_results.push_back(hpx::async<workstealing::DepthPool::workRemaining_action>(victim));
  }

  std::size_t best_idx = 0;
  std::size_t max_jobs = poll_results[0].get();

  for (std::size_t i = 1; i < poll_results.size(); ++i) {
    const auto number_of_jobs = poll_results[i].get();

    if (number_of_jobs > max_jobs) {
      best_idx = i;
      max_jobs = number_of_jobs;
      continue;
    }
  }

  return candidates[best_idx];
}

DepthPoolPolicy::task_t DepthPoolPolicy::stealTaskFrom(hpx::id_type victim) {
  return hpx::async<workstealing::DepthPool::steal_action>(victim).get();
}

void DepthPoolPolicy::updateLastRemoteOnSuccess(hpx::id_type victim) {
  std::unique_lock<mutex_t> l(mtx);
  last_remote = victim;
}

void DepthPoolPolicy::updateLastRemoteOnFailure(hpx::id_type victim, hpx::id_type here) {
  std::unique_lock<mutex_t> l(mtx);
  if (last_remote == victim) {
    last_remote = here;
  }
}

hpx::function<void(), false> DepthPoolPolicy::getWork() {

  // First try to steal from the local pool
  auto task = getLocalTask();
  auto here = hpx::find_here();

  if (task) {
    DepthPoolPolicyPerf::perf_localSteals++;
    return hpx::bind(task, here);
  } else {
    DepthPoolPolicyPerf::perf_failedLocalSteals++;
  }

  // Then try to steal from the last successful remote victim
  hpx::id_type last_victim = here;
  if (tryGetLastRemoteVictim(here, last_victim)) {
    task = stealTaskFrom(last_victim);
    if (task) {
      updateLastRemoteOnSuccess(last_victim);
      DepthPoolPolicyPerf::perf_lastStealSuccesses++;
      DepthPoolPolicyPerf::perf_distributedSteals++;
      return hpx::bind(task, here);
    }

    DepthPoolPolicyPerf::perf_failedDistributedSteals++;
    updateLastRemoteOnFailure(last_victim, here);
  }


  // Last remote failed, try random victims
  auto candidates = chooseDistributedCandidates();
  if (candidates.empty()) {
    return nullptr;
  }

  auto victim = chooseBestVictim(candidates);
  task = stealTaskFrom(victim);
  if (task) {
    updateLastRemoteOnSuccess(victim);
    DepthPoolPolicyPerf::perf_distributedSteals++;
    return hpx::bind(task, here);
  }

  DepthPoolPolicyPerf::perf_failedDistributedSteals++;
  updateLastRemoteOnFailure(victim, here);
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
}

}}
