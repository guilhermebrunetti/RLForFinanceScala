package rl

import rl.FiniteMarkovDecisionProcess.StateActionMapping
import rl.FiniteMarkovProcess.defaultSortingFunction
import rl.FiniteMarkovRewardProcess.{RewardTransition, StateReward}
import rl.utils.{Distribution, FiniteDistribution}

trait HasAction[A] {
  def action: A
}

//trait ActionStep[S, A] extends TransitionStep[S] with HasAction[A]

case class ActionStep[S, A](
  state: NonTerminal[S],
  action: A,
  nextState: State[S],
  reward: Double
) extends TransitionStep[S] with HasAction[A] {
  
  override def addReturn(returns: Double, gamma: Double): ReturnStepMDP[S, A] = {
    ReturnStepMDP(this.state, this.action, this.nextState, this.reward, this.reward + gamma * returns)
  }
}

case class ReturnStepMDP[S, A](
  state: NonTerminal[S],
  action: A,
  nextState: State[S],
  reward: Double,
  returns: Double
) extends ReturnStep[S] with HasAction[A]

trait MarkovDecisionProcess[S, A] {
  self =>
  
  def step(state: NonTerminal[S], action: A): Distribution[(State[S], Double)]
  
  def actions(state: NonTerminal[S]): Iterable[A] // What do we do if actions are continuous?
  
  def applyPolicy(policy: Policy[S, A]): MarkovRewardProcess[S] = {
    
    val rewardProcess = new MarkovRewardProcess[S] {
      override def transitionReward(state: NonTerminal[S]): Distribution[(State[S], Double)] =
        policy
          .act(state)
          .flatMap(action => self.step(state, action))
    }
    
    rewardProcess
  }
  
  def actionTraces(
    initialStateDistribution: Distribution[NonTerminal[S]],
    policy: Policy[S, A]
  ): LazyList[LazyList[ActionStep[S, A]]] = {
    LazyList.continually(simulateAction(initialStateDistribution, policy))
  }
  
  def simulateAction(
    initialStateDistribution: Distribution[NonTerminal[S]],
    policy: Policy[S, A]
  ): LazyList[ActionStep[S, A]] = {
    val initialState = initialStateDistribution.sample
    val action = policy.act(initialState).sample
    val (nextState, reward) = self.step(initialState, action).sample
    val step = ActionStep(initialState, action, nextState, reward)
    
    simulateAction(step, policy)
  }
  
  def simulateAction(
    transitionStep: ActionStep[S, A],
    policy: Policy[S, A]
  ): LazyList[ActionStep[S, A]] = {
    LazyList.unfold(Option(transitionStep))(_.map(nextActionStep(_, policy)))
  }
  
  private def nextActionStep(
    step: ActionStep[S, A],
    policy: Policy[S, A]
  ): (ActionStep[S, A], Option[ActionStep[S, A]]) = {
    def f(nt: NonTerminal[S]): ActionStep[S, A] = {
      val action = policy.act(nt).sample
      val (nextNextState, nextReward) = self.step(nt, action).sample
      ActionStep(nt, action, nextNextState, nextReward)
    }
    
    (step, step.nextState.onNonTerminalOption(f))
  }
  
}

object FiniteMarkovDecisionProcess {
  type ActionTransition[S, A] = Map[A, StateReward[S]]
  type StateActionMapping[S, A] = Map[NonTerminal[S], ActionTransition[S, A]]
  
  def processInputMap[S, A](
    inputMap: Map[S, Map[A, FiniteDistribution[(S, Double)]]]
  ): StateActionMapping[S, A] = {
    val nonTerminalStates = inputMap.keySet
    
    def toState(stateRewardPair: (S, Double)): (State[S], Double) = {
      val (s, reward) = stateRewardPair
      val state = if (nonTerminalStates.contains(s)) NonTerminal(s) else Terminal(s)
      (state, reward)
    }
    
    inputMap.map { case (s, actionMap) =>
      NonTerminal(s) -> actionMap.view.mapValues(_.map(toState)).toMap
    }
  }
}

trait FiniteMarkovDecisionProcess[S, A]
  extends MarkovDecisionProcess[S, A] {
  self =>
  
  def nonTerminalStates: IndexedSeq[NonTerminal[S]] = stateActionMap.keySet.toIndexedSeq.sortWith(stateSortingFunction)
  
  def stateSortingFunction(x: NonTerminal[S], y: NonTerminal[S]): Boolean = defaultSortingFunction(x, y)
  
  def stateActionMap: StateActionMapping[S, A]
  
  override def step(
    state: NonTerminal[S],
    action: A
  ): FiniteDistribution[(State[S], Double)] = stateActionMap(state)(action)
  
  override def actions(state: NonTerminal[S]): Iterable[A] = {
    stateActionMap(state).keys
  }
  
  override def toString: String = {
    stateActionMap
      .toIndexedSeq
      .sortWith { case ((x, _), (y, _)) => stateSortingFunction(x, y) }
      .map { case (state, actionMap) =>
        val prefix = s"From State $state"
        actionMap
          .toIndexedSeq
          .sortWith { case ((x, _), (y, _)) => actionSortingFunction(x, y) }
          .map { case (action, dist) =>
            val suffix = s" with Action $action:\n\t$dist"
            prefix + suffix
          }.mkString("\n")
      }.mkString("\n")
  }
  
  def actionSortingFunction(x: A, y: A): Boolean = defaultSortingFunction(x, y)
  
  def applyFinitePolicy(policy: FinitePolicy[S, A]): FiniteMarkovRewardProcess[S] = {
    
    val rewardProcess = new FiniteMarkovRewardProcess[S] {
      override def transitionRewardMap: RewardTransition[S] = self.stateActionMap.map { case (state, actionMap) =>
        val actions: FiniteDistribution[A] = policy.act(state)
        val outcomes: FiniteDistribution[(State[S], Double)] = actions.flatMap(action => actionMap(action))
        state -> outcomes
      }
    }
    
    rewardProcess
  }
}
  
