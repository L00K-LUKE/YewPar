#ifndef YEWPAR_POLICY_BUFFEREDWORKPOOL_HPP
#define YEWPAR_POLICY_BUFFEREDWORKPOOL_HPP

#include "Policy.hpp"

#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/modules/collectives.hpp>

#include "workstealing/Workqueue.hpp"

#include <random>
#include <vector>
#include <deque>


namespace Workstealing { namespace Scheduler {extern std::shared_ptr<Policy> local_policy; }}

namespace Workstealing { namespace Policies {

namespace BufferedWorkpoolPerf {
void registerPerformanceCounters();
}

class BufferedWorkpool : public Policy {

  private:
    using task_type = hpx::distributed::function<void(hpx::id_type)>;

    static constexpr std::size_t LOCAL_CAPACITY = 256;
    static constexpr std::size_t CHUNK_SIZE = 16;
    static constexpr std::size_t SPILL_LIMIT = 192; // Maybe rename?
    
    hpx::id_type local_workqueue;
    std::vector<hpx::id_type> distributed_workqueues;
    
    struct LocalState {
      std::deque<task_type> local_tasks;     // per-thread buffer
      hpx::id_type last_remote;        // last remote victim (per-thread)
      std::mt19937 rng;                // per-thread RNG
      bool initialized = false;        
    };
    
    static LocalState& tls();
    static void init_tls_if_needed(LocalState& st);

 public:
  BufferedWorkpool(hpx::id_type localQueue);
  ~BufferedWorkpool() = default;

  hpx::function<void(), false> getWork() override;
  void addwork(task_type task);
  void registerDistributedWorkqueues(std::vector<hpx::id_type> workqueues);

  static void setWorkqueue(hpx::id_type localWorkqueue) {
    Workstealing::Scheduler::local_policy = std::make_shared<BufferedWorkpool>(localWorkqueue);
  }
  struct setWorkqueue_act : hpx::actions::make_action<
    decltype(&BufferedWorkpool::setWorkqueue),
    &BufferedWorkpool::setWorkqueue,
    setWorkqueue_act>::type {};

  static void setDistributedWorkqueues(std::vector<hpx::id_type> workqueues) {
    std::static_pointer_cast<Workstealing::Policies::BufferedWorkpool>(Workstealing::Scheduler::local_policy)->registerDistributedWorkqueues(std::move(workqueues));
  }
  struct setDistributedWorkqueues_act : hpx::actions::make_action<
    decltype(&BufferedWorkpool::setDistributedWorkqueues),
    &BufferedWorkpool::setDistributedWorkqueues,
    setDistributedWorkqueues_act>::type {};

  static void initPolicy() {
    std::vector<hpx::future<void> > futs;
    std::vector<hpx::id_type> workqueues;
    for (auto const& loc : hpx::find_all_localities()) {
      auto workqueue = hpx::new_<workstealing::Workqueue>(loc).get();
      futs.push_back(hpx::async<setWorkqueue_act>(loc, workqueue));
      workqueues.push_back(workqueue);
    }
    hpx::wait_all(futs);
    hpx::wait_all(hpx::lcos::broadcast<setDistributedWorkqueues_act>(hpx::find_all_localities(), workqueues));
  }
};

}}

#endif
