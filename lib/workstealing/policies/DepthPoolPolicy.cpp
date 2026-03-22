#include "DepthPoolPolicy.hpp"

#include <hpx/functional/function.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

#include <algorithm>
#include <memory>

#include "util/util.hpp"

namespace Workstealing { namespace Policies {

std::atomic<bool> DepthPoolPolicy::use_last_steal{true};
std::atomic<std::size_t> DepthPoolPolicy::random_steal_attempts{2};
std::atomic<std::size_t> DepthPoolPolicy::lifeline_degree{2};

namespace DepthPoolPolicyPerf {

std::atomic<std::uint64_t> perf_spawns(0);
std::atomic<std::uint64_t> perf_localSteals(0);
std::atomic<std::uint64_t> perf_distributedSteals(0);
std::atomic<std::uint64_t> perf_failedLocalSteals(0);
std::atomic<std::uint64_t> perf_failedDistributedSteals(0);
std::atomic<std::uint64_t> perf_lifelineModeEntries(0);
std::atomic<std::uint64_t> perf_lifelineSteals(0);
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
std::uint64_t getLifelineModeEntries(bool reset) { return get_and_reset(perf_lifelineModeEntries, reset);}
std::uint64_t getLifelineSteals(bool reset) { return get_and_reset(perf_lifelineSteals, reset);}
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
      "/workstealing/depthpool/lifelineModeEntries",
      &getLifelineModeEntries,
      "Returns the number of times this locality enters lifeline mode"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/lifelineSteals",
      &getLifelineSteals,
      "Returns the number of successful steals from lifeline victims"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/lastStealSuccesses",
      &getlastStealSuccesses,
      "Returns the number of successful steals from the previous successful victim"
                                                  );
}

}

DepthPoolPolicy::DepthPoolPolicy(hpx::id_type workpool) {
  local_workpool = workpool;
  last_remote = hpx::find_here();

  std::random_device rd;
  randGenerator.seed(rd());
}

DepthPoolPolicy::RemoteStealState
DepthPoolPolicy::prepareRemoteStealState(hpx::id_type const& here) {
  RemoteStealState state;
  state.preferred_victim = here;

  std::unique_lock<mutex_t> l(mtx);
  state.random_victims = distributed_workpools;
  state.lifeline_victims = lifeline_workpools;
  state.preferred_victim = last_remote;
  state.try_last_victim =
    use_last_steal.load(std::memory_order_relaxed) && state.preferred_victim != here;

  if (state.try_last_victim) {
    state.random_victims.erase(
      std::remove(state.random_victims.begin(), state.random_victims.end(), state.preferred_victim),
      state.random_victims.end());
  }

  state.random_attempts = std::min(
    random_steal_attempts.load(std::memory_order_relaxed),
    state.random_victims.size());
  return state;
}

bool DepthPoolPolicy::hasRemoteCandidates(RemoteStealState const& state) const {
  return state.try_last_victim || !state.random_victims.empty() || !state.lifeline_victims.empty();
}

bool DepthPoolPolicy::inLifelineMode() {
  std::unique_lock<mutex_t> l(mtx);
  return lifeline_mode;
}

void DepthPoolPolicy::setLifelineMode(bool enabled) {
  std::unique_lock<mutex_t> l(mtx);
  if (!lifeline_mode && enabled) {
    DepthPoolPolicyPerf::perf_lifelineModeEntries++;
  }
  lifeline_mode = enabled;
}

std::size_t DepthPoolPolicy::drawRandomVictimIndex(std::size_t victim_count) {
  std::unique_lock<mutex_t> l(mtx);
  std::uniform_int_distribution<std::size_t> rand(0, victim_count - 1);
  return rand(randGenerator);
}

bool DepthPoolPolicy::tryRemoteVictim(
  hpx::id_type const& victim,
  hpx::distributed::function<void(hpx::id_type)> & task) {
  task = hpx::async<workstealing::DepthPool::steal_action>(victim).get();
  if (!task) {
    DepthPoolPolicyPerf::perf_failedDistributedSteals++;
    return false;
  }

  {
    std::unique_lock<mutex_t> l(mtx);
    last_remote = victim;
  }
  DepthPoolPolicyPerf::perf_distributedSteals++;
  return true;
}

void DepthPoolPolicy::forgetPreferredVictimIfUnchanged(
  hpx::id_type const& preferred_victim,
  hpx::id_type const& here) {
  std::unique_lock<mutex_t> l(mtx);
  if (last_remote == preferred_victim) {
    last_remote = here;
  }
}

bool DepthPoolPolicy::tryPreferredVictim(
  RemoteStealState const& state,
  hpx::id_type const& here,
  std::vector<hpx::id_type> & attempted_victims,
  hpx::distributed::function<void(hpx::id_type)> & task) {
  if (!state.try_last_victim) {
    return false;
  }

  if (tryRemoteVictim(state.preferred_victim, task)) {
    DepthPoolPolicyPerf::perf_lastStealSuccesses++;
    return true;
  }

  attempted_victims.push_back(state.preferred_victim);
  forgetPreferredVictimIfUnchanged(state.preferred_victim, here);
  return false;
}

bool DepthPoolPolicy::tryRandomVictims(
  RemoteStealState & state,
  std::vector<hpx::id_type> & attempted_victims,
  hpx::distributed::function<void(hpx::id_type)> & task) {
  for (std::size_t i = 0; i < state.random_attempts && !state.random_victims.empty(); ++i) {
    auto victim_index = drawRandomVictimIndex(state.random_victims.size());
    auto victim = state.random_victims[victim_index];

    state.random_victims[victim_index] = state.random_victims.back();
    state.random_victims.pop_back();

    if (tryRemoteVictim(victim, task)) {
      return true;
    }
    attempted_victims.push_back(victim);
  }
  return false;
}

bool DepthPoolPolicy::tryLifelineVictims(
  RemoteStealState const& state,
  std::vector<hpx::id_type> & attempted_victims,
  hpx::distributed::function<void(hpx::id_type)> & task) {
  for (auto const& victim : state.lifeline_victims) {
    if (std::find(attempted_victims.begin(), attempted_victims.end(), victim) != attempted_victims.end()) {
      continue;
    }

    if (tryRemoteVictim(victim, task)) {
      DepthPoolPolicyPerf::perf_lifelineSteals++;
      return true;
    }
    attempted_victims.push_back(victim);
  }
  return false;
}

hpx::function<void(), false> DepthPoolPolicy::getWork() {

  auto here = hpx::find_here();
  hpx::distributed::function<void(hpx::id_type)> task;

  // Local steal attempt.
  task = hpx::async<workstealing::DepthPool::getLocal_action>(local_workpool).get();
  if (task) {
    setLifelineMode(false);
    DepthPoolPolicyPerf::perf_localSteals++;
    return hpx::bind(task, here);
  }
  DepthPoolPolicyPerf::perf_failedLocalSteals++;

  // Lifeline mode skips remote random steals.
  if (inLifelineMode()) {
    auto state = prepareRemoteStealState(here);
    if (!state.lifeline_victims.empty()) {
      std::vector<hpx::id_type> attempted_victims;
      attempted_victims.reserve(state.lifeline_victims.size());
      if (tryLifelineVictims(state, attempted_victims, task)) {
        setLifelineMode(false);
        return hpx::bind(task, here);
      }
      return nullptr;
    }

    // No lifelines available means lifeline-only mode cannot make progress.
    setLifelineMode(false);
  }


  // Remote steal attempts.
  auto state = prepareRemoteStealState(here);
  if (!hasRemoteCandidates(state)) {
    return nullptr;
  }

  std::vector<hpx::id_type> attempted_victims;
  attempted_victims.reserve(1 + state.random_attempts + state.lifeline_victims.size());
  if (tryPreferredVictim(state, here, attempted_victims, task) ||
      tryRandomVictims(state, attempted_victims, task)) {
    setLifelineMode(false);
    return hpx::bind(task, here);
  }

  // Remote phase failed: enter lifeline mode.
  if (!state.lifeline_victims.empty()) {
    setLifelineMode(true);
    if (tryLifelineVictims(state, attempted_victims, task)) {
      setLifelineMode(false);
      return hpx::bind(task, here);
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
  auto local_it = std::find_if(
    workpools.begin(),
    workpools.end(),
    YewPar::util::isColocated);
  auto local_idx = static_cast<std::size_t>(std::distance(workpools.begin(), local_it));

  distributed_workpools.erase(
      std::remove_if(distributed_workpools.begin(), distributed_workpools.end(), YewPar::util::isColocated),
      distributed_workpools.end());

  lifeline_workpools.clear();
  if (workpools.size() <= 1 || local_it == workpools.end()) {
    return;
  }

  const auto available = workpools.size() - 1;
  const auto degree = std::min(
    lifeline_degree.load(std::memory_order_relaxed),
    available);

  if (degree == 0) {
    return;
  }

  lifeline_workpools.reserve(degree);
  auto add_lifeline = [&](std::size_t idx) {
    if (idx == local_idx || idx >= workpools.size()) {
      return;
    }
    auto victim = workpools[idx];
    if (std::find(lifeline_workpools.begin(), lifeline_workpools.end(), victim) != lifeline_workpools.end()) {
      return;
    }
    lifeline_workpools.push_back(victim);
  };

  // Start with power-of-two neighbours to approximate a sparse lifeline graph.
  for (std::size_t hop = 1; hop < workpools.size() && lifeline_workpools.size() < degree; hop <<= 1) {
    add_lifeline((local_idx + hop) % workpools.size());
  }

  // Fill remaining slots by scanning clockwise.
  for (std::size_t hop = 1; hop < workpools.size() && lifeline_workpools.size() < degree; ++hop) {
    add_lifeline((local_idx + hop) % workpools.size());
  }
}

}}
