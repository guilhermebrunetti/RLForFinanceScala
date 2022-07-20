package rl

import rl.DynamicProgramming.{ValueFunction, extendedValueFunction}
import rl.FiniteMarkovDecisionProcess.{ActionTransition, StateActionMapping}
import rl.FiniteMarkovRewardProcess.{RewardTransition, StateReward}
import rl.utils.FiniteDistribution

case class WithTime[S](state: S, time: Int = 0) {
  /**
   * A wrapper that augments a state of type S with a time field.
   */
  def stepTime: WithTime[S] = this.copy[S](time = this.time + 1)
  def stepTime(newState: S): WithTime[S] = this.copy[S](state = newState, time = this.time + 1)
  def pair: (S, Int) = (state, time)
  def step(f: (S, Int) => S): WithTime[S] = WithTime(
    state = f(this.state, this.time),
    time = this.time + 1
  )
}


object FiniteHorizon {
  type RewardOutcome[S] = FiniteDistribution[(WithTime[S], Double)]
  
  /**
   * Turn a normal FiniteMarkovRewardProcess into one with a finite horizon
   * that stops after 'limit' steps.
   * Note that this makes the data representation of the process
   * larger, since we end up having distinct sets and transitions for
   * every single time step up to the limit.
   */
  def finiteHorizonMarkovRewardProcess[S](
    mrp: FiniteMarkovRewardProcess[S],
    limit: Int
  ): FiniteMarkovRewardProcess[WithTime[S]] = {
    
    val inputMap: Map[WithTime[S], FiniteDistribution[(WithTime[S], Double)]] =
      (0 until limit).flatMap { time =>
        mrp.nonTerminalStates.map { state =>
          val distribution: FiniteDistribution[(State[S], Double)] = mrp.transitionReward(state)
          val stateWithTime: WithTime[S] = WithTime(state.state, time)
          val distributionWithTime: FiniteDistribution[(WithTime[S], Double)] =
            distribution.map { case (nextState, reward) =>
              (WithTime(nextState.state, time + 1), reward)
            }
          stateWithTime -> distributionWithTime
        }
      }.toMap
    
    new FiniteMarkovRewardProcess[WithTime[S]] {
      override def transitionRewardMap: RewardTransition[WithTime[S]] =
        FiniteMarkovRewardProcess.processInputMap(inputMap)
    }
  }
  
  /**
   * Given a finite-horizon process, break the transition between each
   * time step (starting with 0) into its own data structure. This
   * representation makes it easier to implement backwards
   * induction.
   */
  def unwrapFiniteHorizonMRP[S](
    process: FiniteMarkovRewardProcess[WithTime[S]]
  ): Seq[RewardTransition[S]] = {
    
    def withoutTime(arg: StateReward[WithTime[S]]): StateReward[S] =
      arg.map(singleWithoutTime)
    
    val nonTerminalState: Seq[NonTerminal[WithTime[S]]] = process.nonTerminalStates
    val byTime: Map[Int, Seq[NonTerminal[WithTime[S]]]] = nonTerminalState.groupBy(x => x.state.time)
    val seqTimeRewards: Seq[(Int, Map[NonTerminal[S], StateReward[S]])] = byTime
      .view
      .mapValues { stateSeq =>
        stateSeq.map { nt =>
          NonTerminal(nt.state.state) -> withoutTime(process.transitionReward(nt))
        }.toMap
      }.toSeq
    
    seqTimeRewards
      .sortBy { case (time, _) => time }
      .map(_._2)
  }
  
  protected def singleWithoutTime[S](
    stateReward: (State[WithTime[S]], Double)
  ): (State[S], Double) = {
    val (stateWithTime, reward) = stateReward
    (stateWithTime.map(_.state), reward)
  }
  
  /**
   * Evaluate the given finite Markov reward process using backwards
   * induction, given that the process stops after limit time steps.
   */
  def evaluate[S](
    steps: Seq[RewardTransition[S]],
    gamma: Double
  ): Seq[ValueFunction[S]] = {
    steps.lastOption.map { last =>
      val initial = steps.init
      val finalValueFunction = last
        .view
        .mapValues { rewardTransition =>
          rewardTransition.expectation { case (_, reward) => reward }
        }.toMap
      initial.scanRight(finalValueFunction) { case (rewardMap, valueFunction) =>
        rewardMap
          .view
          .mapValues { rewardTransition =>
            rewardTransition.expectation { case (nextState, reward) =>
              reward + gamma * extendedValueFunction(valueFunction)(nextState)
            }
          }.toMap
      }
    }.getOrElse(Seq.empty)
  }
  
  /**
   * Turn a normal FiniteMarkovRewardProcess into one with a finite horizon
   * that stops after 'limit' steps.
   * Note that this makes the data representation of the process
   * larger, since we end up having distinct sets and transitions for
   * every single time step up to the limit.
   */
  def finiteHorizonMarkovDecisionProcess[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    limit: Int
  ): FiniteMarkovDecisionProcess[WithTime[S], A] = {
    
    val inputMap: Map[WithTime[S], Map[A, FiniteDistribution[(WithTime[S], Double)]]] =
      (0 until limit).flatMap { time =>
        mdp.nonTerminalStates.map { state =>
          val actionDistribution = mdp.stateActionMap(state)
          val stateWithTime: WithTime[S] = WithTime(state.state, time)
          val actionDistributionWithTime = actionDistribution
            .view
            .mapValues { distribution =>
              distribution.map { case (nextState, reward) =>
                (WithTime(nextState.state, time + 1), reward)
              }
            }.toMap
          
          stateWithTime -> actionDistributionWithTime
        }
      }.toMap
    
    new FiniteMarkovDecisionProcess[WithTime[S], A] {
      
      override def stateSortingFunction(
        x: NonTerminal[WithTime[S]],
        y: NonTerminal[WithTime[S]]
      ): Boolean = if (x.state.time == y.state.time) {
        mdp.stateSortingFunction(x.map(_.state), y.map(_.state))
      } else x.state.time < y.state.time
  
      override def actionSortingFunction(x: A, y: A): Boolean = mdp.actionSortingFunction(x, y)
      
      override def stateActionMap: StateActionMapping[WithTime[S], A] =
        FiniteMarkovDecisionProcess.processInputMap(inputMap)
    }
  }
  
  /**
   * Unwrap a finite Markov decision process into a sequence of
   * transitions between each time step (starting with 0). This
   * representation makes it easier to implement backwards induction.
   */
  def unwrapFiniteHorizonMDP[S, A](
    process: FiniteMarkovDecisionProcess[WithTime[S], A]
  ): Seq[StateActionMapping[S, A]] = {
    
    
    def withoutTime(arg: StateActionMapping[WithTime[S], A]): StateActionMapping[S, A] =
      arg.map { case (nonTerminal, actionTransition) =>
        nonTerminal.map(_.state) ->
          actionTransition.view.mapValues(_.map(singleWithoutTime)).toMap
      }
    
    val nonTerminalState: Seq[NonTerminal[WithTime[S]]] = process.nonTerminalStates
    val byTime: Map[Int, Seq[NonTerminal[WithTime[S]]]] = nonTerminalState.groupBy(x => x.state.time)
    val seqTimeRewards: Seq[(Int, StateActionMapping[S, A])] = byTime
      .view
      .mapValues { stateSeq =>
        withoutTime(
          stateSeq.map { nt =>
            nt -> process.stateActionMap(nt)
          }.toMap
        )
      }.toSeq
    
    seqTimeRewards
      .sortBy { case (time, _) => time }
      .map(_._2)
  }
  
  /**
   * Use backwards induction to find the optimal value function and optimal
   * policy at each time step
   */
  def optimalValueFunctionAndPolicy[S, A](
    steps: Seq[StateActionMapping[S, A]],
    gamma: Double
  ): Seq[(ValueFunction[S], FiniteDeterministicPolicy[S, A])] = {
    steps.lastOption.map { last =>
      val initial = steps.init
      val finalQValues: Map[NonTerminal[S], (A, Double)] = last
        .view
        .mapValues { actionTransition =>
          actionTransition
            .view
            .mapValues(_.expectation { case (_, reward) => reward })
            .toMap
            .maxBy { case (_, v) => v }
        }.toMap
      
      val (finalValueFunction, finalPolicy) = valueFunctionAndPolicyFromQValues(finalQValues)
      
      initial.scanRight((finalValueFunction, finalPolicy)) { case (stateActionMapping, (valueFunction, _)) =>
        val qValues: Map[NonTerminal[S], (A, Double)] = stateActionMapping
          .view
          .mapValues { actionTransition =>
            actionTransition
              .view
              .mapValues(_.expectation { case (nextState, reward) =>
                reward + gamma * extendedValueFunction(valueFunction)(nextState)
              })
              .toMap
              .maxBy { case (_, v) => v }
          }.toMap
        valueFunctionAndPolicyFromQValues(qValues)
      }
    }.getOrElse(Seq.empty)
  }
  
  protected def valueFunctionAndPolicyFromQValues[S, A](
    qValues: Map[NonTerminal[S], (A, Double)]
  ): (ValueFunction[S], FiniteDeterministicPolicy[S, A]) = {
    
    val finalValueFunction = qValues
      .view
      .mapValues(_._2)
      .toMap
    
    val finalPolicy = FiniteDeterministicPolicy(qValues
      .map { case (state, (action, _)) => state.state -> action }
    )
    
    (finalValueFunction, finalPolicy)
  }
}
