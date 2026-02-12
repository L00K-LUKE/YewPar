#ifndef YEWPAR_POLICY_DEPTHPOOL_HPP
#define YEWPAR_POLICY_DEPTHPOOL_HPP

#include "Policy.hpp"

#include <hpx/include/components.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>

#include "../DepthPool.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace Workstealing { namespace Scheduler {extern std::shared_ptr<Policy> local_policy; }}

namespace Workstealing { namespace Policies {

namespace DepthPoolPolicyPerf {
void registerPerformanceCounters();
}


// TODO: Performance counters
class DepthPoolPolicy : public Policy {

 private:
  hpx::id_type local_workpool;
  hpx::id_type last_remote;
  std::vector<hpx::id_type> distributed_workpools;
  unsigned neighbour_steal_size;

  // random number generator
  std::mt19937 randGenerator;

  using mutex_t = hpx::mutex;
  mutex_t mtx;

 public:
  DepthPoolPolicy(hpx::id_type workpool, unsigned neighbourStealSize);
  ~DepthPoolPolicy() = default;

  hpx::function<void(), false> getWork() override;

  void addwork(hpx::distributed::function<void(hpx::id_type)> task, unsigned depth);

  void registerDistributedDepthPools(std::vector<hpx::id_type> workpools);

  static void setDepthPool(hpx::id_type localworkpool, unsigned neighbourStealSize) {
    Workstealing::Scheduler::local_policy = std::make_shared<DepthPoolPolicy>(localworkpool, neighbourStealSize);
  }
  struct setDepthPool_act : hpx::actions::make_action<
    decltype(&DepthPoolPolicy::setDepthPool),
    &DepthPoolPolicy::setDepthPool,
    setDepthPool_act>::type {};

  static void setDistributedDepthPools(std::vector<hpx::id_type> workpools) {
    std::static_pointer_cast<Workstealing::Policies::DepthPoolPolicy>(Workstealing::Scheduler::local_policy)->registerDistributedDepthPools(workpools);
  }
  struct setDistributedDepthPools_act : hpx::actions::make_action<
    decltype(&DepthPoolPolicy::setDistributedDepthPools),
    &DepthPoolPolicy::setDistributedDepthPools,
    setDistributedDepthPools_act>::type {};

  static void initPolicy() {
    auto const neighbourStealSize =
        static_cast<unsigned>(std::stoul(hpx::get_config_entry("hpx.ring_size", "1")));

    std::vector<hpx::future<void> > futs;
    std::vector<hpx::id_type> pools;
    for (auto const& loc : hpx::find_all_localities()) {
      auto depthpool = hpx::new_<workstealing::DepthPool>(loc).get();
      futs.push_back(hpx::async<setDepthPool_act>(loc, depthpool, neighbourStealSize));
      pools.push_back(depthpool);
    }
    hpx::wait_all(futs);
    hpx::wait_all(hpx::lcos::broadcast<setDistributedDepthPools_act>(hpx::find_all_localities(), pools));
  }
};

}}

#endif
