#include "BufferedWorkpool.hpp"

#include <hpx/functional/function.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

#include <memory>
#include <algorithm>

#include "util/util.hpp"

namespace Workstealing { namespace Policies {

namespace BufferedWorkpoolPerf {

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
      "/workstealing/BufferedWorkpool/spawns",
      &getSpawns,
      "Returns the number of tasks spawned on this locality"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/BufferedWorkpool/localSteals",
      &getLocalSteals,
      "Returns the number of tasks taken from locality queue"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/BufferedWorkpool/distributedSteals",
      &getDistributedSteals,
      "Returns the number of tasks stolen from another thread on another locality"
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/BufferedWorkpool/localFailedSteals",
      &getFailedLocalSteals,
      "Returns the number of failed steals from this locality "
                                                  );

  hpx::performance_counters::install_counter_type(
      "/workstealing/BufferedWorkpool/distributedFailedSteals",
      &getFailedDistributedSteals,
      "Returns the number of failed steals from another locality "
                                                  );

}

}


void BufferedWorkpool::init_tls_if_needed(BufferedWorkpool::LocalState& st)
{
  if (st.initialized) return;

  st.last_remote = hpx::find_here();

  std::random_device rd;
  std::seed_seq seed{ rd(), rd(), rd(), rd() };
  st.rng.seed(seed);

  st.initialized = true;
}


BufferedWorkpool::LocalState& BufferedWorkpool::tls() {
  static thread_local LocalState st;
  return st;
}

BufferedWorkpool::BufferedWorkpool(hpx::id_type localQueue) : 
local_workqueue(std::move(localQueue)) 
{}

hpx::function<void(), false> BufferedWorkpool::getWork()
{
  auto& st = tls();
  init_tls_if_needed(st);

  task_type task;

  // Try to get locally
  if (!st.local_tasks.empty()) {
    task_type task = std::move(st.local_tasks.back());
    st.local_tasks.pop_back();
    return hpx::bind(std::move(task), hpx::find_here());
  }

  // No local tasks, so refil local buffer with chunks from the locality workqueue
  for (std::size_t i = 0; i < CHUNK_SIZE && st.local_tasks.size() < LOCAL_CAPACITY; i++) {
    task_type task =
      hpx::async<workstealing::Workqueue::getLocal_action>(local_workqueue).get();
    
    if (!task) break;

    BufferedWorkpoolPerf::perf_localSteals++;
    st.local_tasks.push_back(std::move(task));
  }

  // After steal, try again
  if (!st.local_tasks.empty()) {
    task_type t = std::move(st.local_tasks.back());
    st.local_tasks.pop_back();
    return hpx::bind(std::move(t), hpx::find_here());
  }

  BufferedWorkpoolPerf::perf_failedLocalSteals++;
  // Distributed steal
  if (!distributed_workqueues.empty()) {

      // Try last victim first
      if (st.last_remote != hpx::find_here()) {
          task = hpx::async<workstealing::Workqueue::steal_action>(st.last_remote).get();
          if (task) {
              BufferedWorkpoolPerf::perf_distributedSteals++;
              return hpx::bind(std::move(task), hpx::find_here());
          }
          BufferedWorkpoolPerf::perf_failedDistributedSteals++;
          st.last_remote = hpx::find_here();
      }

      // Random victim
      std::uniform_int_distribution<std::size_t> dist(0, distributed_workqueues.size() - 1);
      const auto victim = distributed_workqueues[dist(st.rng)];

      task = hpx::async<workstealing::Workqueue::steal_action>(victim).get();
      if (task) {
          st.last_remote = victim;
          BufferedWorkpoolPerf::perf_distributedSteals++;
          return hpx::bind(std::move(task), hpx::find_here());
      }
      BufferedWorkpoolPerf::perf_failedDistributedSteals++;
  }

  return nullptr;
}


void BufferedWorkpool::addwork(task_type task) {
  auto& st = tls();
  init_tls_if_needed(st);

  BufferedWorkpoolPerf::perf_spawns++;

  // Enqueue Locality first
  st.local_tasks.push_back(std::move(task));

  // Spill if buffer is getting too big
  if (st.local_tasks.size() > SPILL_LIMIT) {
    const std::size_t spill_count =
      std::min<std::size_t>( CHUNK_SIZE, st.local_tasks.size());

    for (std::size_t i = 0; i < spill_count; i++) {
      task_type task = std::move(st.local_tasks.front());
      st.local_tasks.pop_front();
      hpx::post<workstealing::Workqueue::addWork_action>(local_workqueue, std::move(task));
    }
  }
}

void BufferedWorkpool::registerDistributedWorkqueues(std::vector<hpx::id_type> workqueues)
{
    distributed_workqueues = std::move(workqueues);
    distributed_workqueues.erase(
        std::remove_if(distributed_workqueues.begin(),
                       distributed_workqueues.end(),
                       YewPar::util::isColocated),
        distributed_workqueues.end());
}


}}
