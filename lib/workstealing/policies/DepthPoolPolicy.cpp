#include "DepthPoolPolicy.hpp"

#include <hpx/async_colocated/get_colocation_id.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "util/util.hpp"

namespace Workstealing { namespace Policies {

namespace {

std::uint64_t median(std::vector<std::uint64_t>& samples) {
  if (samples.empty()) {
    return 0;
  }

  auto mid = samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2);
  std::nth_element(samples.begin(), mid, samples.end());
  return *mid;
}

std::uint64_t probeLocalityRttNs(const hpx::id_type& locality, unsigned samples = 3) {
  std::vector<std::uint64_t> rtt_samples;
  rtt_samples.reserve(samples);

  for (unsigned i = 0; i < samples; ++i) {
    auto start = std::chrono::steady_clock::now();
    hpx::async<DepthPoolPolicy::ping_act>(locality).get();
    auto stop = std::chrono::steady_clock::now();

    auto rtt = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    rtt_samples.push_back(static_cast<std::uint64_t>(rtt));
  }

  return median(rtt_samples);
}

}

std::atomic<bool> DepthPoolPolicy::use_last_steal{true};

namespace DepthPoolPolicyPerf {

std::atomic<std::uint64_t> perf_spawns(0);
std::atomic<std::uint64_t> perf_localSteals(0);
std::atomic<std::uint64_t> perf_distributedSteals(0);
std::atomic<std::uint64_t> perf_distributedNearSteals(0);
std::atomic<std::uint64_t> perf_distributedMidSteals(0);
std::atomic<std::uint64_t> perf_distributedFarSteals(0);
std::atomic<std::uint64_t> perf_lastStealTriggers(0);
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
std::uint64_t getLastStealTriggers(bool reset) { return get_and_reset(perf_lastStealTriggers, reset);}
std::uint64_t getFailedLocalSteals(bool reset) { return get_and_reset(perf_failedLocalSteals, reset);}
std::uint64_t getFailedDistributedSteals(bool reset) { return get_and_reset(perf_failedDistributedSteals, reset);}
std::uint64_t getDistributedNearSteals (bool reset) { return get_and_reset(perf_distributedNearSteals, reset);}
std::uint64_t getDistributedMidSteals (bool reset) { return get_and_reset(perf_distributedMidSteals, reset);}
std::uint64_t getDistributedFarSteals (bool reset) { return get_and_reset(perf_distributedFarSteals, reset);}

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
      "/workstealing/depthpool/lastStealTriggers",
      &getLastStealTriggers,
      "Returns the number of times the last-steal optimisation was attempted"
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
      "/workstealing/depthpool/distributedNearSteals",
      &getDistributedNearSteals,
      "Returns the number of successful distributed steals from near localities"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/distributedMidSteals",
      &getDistributedMidSteals,
      "Returns the number of successful distributed steals from mid-distance localities"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/depthpool/distributedFarSteals",
      &getDistributedFarSteals,
      "Returns the number of successful distributed steals from far localities"
                                                  );
}

}

DepthPoolPolicy::DepthPoolPolicy(hpx::id_type workpool) {
  local_workpool = workpool;
  last_remote = hpx::find_here();

  std::random_device rd;
  randGenerator.seed(rd());
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
  std::size_t near_start = 0;
  std::size_t mid_start = 0;
  std::size_t far_start = 0;
  bool has_remote = false;

  {
    std::unique_lock<mutex_t> l(mtx);
    if (!distributed_workpools_by_rtt.empty()) {
      preferred_victim = last_remote;
      has_remote = true;

      if (!near_workpools.empty()) {
        std::uniform_int_distribution<std::size_t> rand(0, near_workpools.size() - 1);
        near_start = rand(randGenerator);
      }
      if (!mid_workpools.empty()) {
        std::uniform_int_distribution<std::size_t> rand(0, mid_workpools.size() - 1);
        mid_start = rand(randGenerator);
      }
      if (!far_workpools.empty()) {
        std::uniform_int_distribution<std::size_t> rand(0, far_workpools.size() - 1);
        far_start = rand(randGenerator);
      }
    }
  }

  if (has_remote) {
    // Last steal optimisation
    if (use_last_steal.load(std::memory_order_relaxed) && preferred_victim != here) {
      task = hpx::async<workstealing::DepthPool::steal_action>(preferred_victim).get();
      if (task) {
        DepthPoolPolicyPerf::perf_lastStealTriggers++;
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

    auto try_tier = [&] (std::vector<hpx::id_type> const& victims, std::size_t start_idx) -> hpx::function<void(), false> {
      if (victims.empty()) {
        return nullptr;
      }

      for (std::size_t offset = 0; offset < victims.size(); ++offset) {
        auto const victim = victims[(start_idx + offset) % victims.size()];
        task = hpx::async<workstealing::DepthPool::steal_action>(victim).get();
        if (task) {
          std::unique_lock<mutex_t> l(mtx);
          last_remote = victim;
          DepthPoolPolicyPerf::perf_distributedSteals++;
          return hpx::bind(task, here);
        }

        DepthPoolPolicyPerf::perf_failedDistributedSteals++;
      }
      return nullptr;
    };

    if (auto stolen = try_tier(near_workpools, near_start)) {
      DepthPoolPolicyPerf::perf_distributedNearSteals++;
      return stolen;
    }
    if (auto stolen = try_tier(mid_workpools, mid_start)) {
      DepthPoolPolicyPerf::perf_distributedMidSteals++;
      return stolen;
    }
    if (auto stolen = try_tier(far_workpools, far_start)) {
      DepthPoolPolicyPerf::perf_distributedFarSteals++;
      return stolen;
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
  workpools.erase(
      std::remove_if(workpools.begin(), workpools.end(), YewPar::util::isColocated),
      workpools.end());

  std::vector<RemoteDepthPool> remote_workpools;
  remote_workpools.reserve(workpools.size());

  for (auto const& pool : workpools) {
    auto locality = hpx::get_colocation_id(hpx::launch::sync, pool);
    auto rtt_ns = probeLocalityRttNs(locality);
    remote_workpools.push_back({pool, locality, rtt_ns});
  }

  std::sort(
      remote_workpools.begin(),
      remote_workpools.end(),
      [] (RemoteDepthPool const& a, RemoteDepthPool const& b) {
        return a.median_rtt_ns < b.median_rtt_ns;
      });

  std::vector<hpx::id_type> ordered_pool_ids;
  ordered_pool_ids.reserve(remote_workpools.size());
  for (auto const& remote : remote_workpools) {
    ordered_pool_ids.push_back(remote.pool);
  }

  std::vector<hpx::id_type> near_pools;
  std::vector<hpx::id_type> mid_pools;
  std::vector<hpx::id_type> far_pools;

  auto const n = remote_workpools.size();
  auto const t0 = (n + 2) / 3;
  auto const t1 = (2 * n + 2) / 3;
  near_pools.reserve(t0);
  mid_pools.reserve(t1 - t0);
  far_pools.reserve(n - t1);

  for (std::size_t i = 0; i < n; ++i) {
    auto const& pool = remote_workpools[i].pool;
    if (i < t0) {
      near_pools.push_back(pool);
    } else if (i < t1) {
      mid_pools.push_back(pool);
    } else {
      far_pools.push_back(pool);
    }
  }

  std::unique_lock<mutex_t> l(mtx);
  distributed_workpools = std::move(ordered_pool_ids);
  distributed_workpools_by_rtt = std::move(remote_workpools);
  near_workpools = std::move(near_pools);
  mid_workpools = std::move(mid_pools);
  far_workpools = std::move(far_pools);
}

}}
