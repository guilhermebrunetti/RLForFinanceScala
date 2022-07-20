package rl

import breeze.linalg._
import rl.FiniteMarkovProcess.Transition
import rl.FiniteMarkovRewardProcess.RewardTransition
import rl.utils.{Distribution, FiniteDistribution}

trait StateTransition[S] {
  def state: NonTerminal[S]
  
  def nextState: State[S]
  
  def byState: S = state.state
}

trait HasReward {
  def reward: Double
}

trait HasReturn {
  def returns: Double
}

trait TransitionStep[S] extends StateTransition[S] with HasReward {
  def addReturn(returns: Double, gamma: Double): ReturnStep[S]
  
  def nextStateAndReward: (State[S], Double) = (nextState, reward)
}

trait ReturnStep[S] extends StateTransition[S] with HasReward with HasReturn

final case class TransitionStepMRP[S](
  state: NonTerminal[S],
  nextState: State[S],
  reward: Double
) extends TransitionStep[S] {
  
  override def addReturn(returns: Double, gamma: Double): ReturnStepMRP[S] =
    ReturnStepMRP(this.state, this.nextState, this.reward, this.reward + gamma * returns)
}

final case class ReturnStepMRP[S](
  state: NonTerminal[S],
  nextState: State[S],
  reward: Double,
  returns: Double
) extends ReturnStep[S]

trait MarkovRewardProcess[S] extends MarkovProcess[S] {
  
  /**
   * Given a state, returns a distribution of the next state
   * and reward from transitioning between the states.
   */
  def transitionReward(state: NonTerminal[S]): Distribution[(State[S], Double)]
  
  override def transition(state: NonTerminal[S]): Distribution[State[S]] = {
    transitionReward(state).map { case (nextState, _) => nextState }
  }
  
  def rewardTraces(
    initialStateDistribution: Distribution[NonTerminal[S]]
  ): LazyList[LazyList[TransitionStep[S]]] = {
    LazyList.continually(simulateReward(initialStateDistribution))
  }
  
  def simulateReward(
    initialStateDistribution: Distribution[NonTerminal[S]]
  ): LazyList[TransitionStep[S]] = {
    val initialState = initialStateDistribution.sample
    val (nextState, reward) = transitionReward(initialState).sample
    val transitionStep = TransitionStepMRP(initialState, nextState, reward)
    simulateReward(transitionStep)
  }
  
  def simulateReward(
    transitionStep: TransitionStep[S]
  ): LazyList[TransitionStep[S]] = {
    LazyList.unfold(Option(transitionStep))(_.map(nextTransitionStep))
  }
  
  private def nextTransitionStep(step: TransitionStep[S]): (TransitionStep[S], Option[TransitionStep[S]]) = {
    def f(nt: NonTerminal[S]): TransitionStep[S] = {
      val (nextNextState, nextReward) = transitionReward(nt).sample
      TransitionStepMRP(nt, nextNextState, nextReward)
    }
    
    (step, step.nextState.onNonTerminalOption(f))
  }
  
}

trait FiniteMarkovRewardProcess[S]
  extends MarkovRewardProcess[S] with FiniteMarkovProcess[S] {
  
  def transitionRewardMap: RewardTransition[S]
  
  override def transitionReward(state: NonTerminal[S]): FiniteDistribution[(State[S], Double)] = {
    transitionRewardMap(state)
  }
  
  override def transitionMap: Transition[S] = {
    transitionRewardMap
      .view
      .mapValues { finiteDistribution =>
        finiteDistribution.map { case (s, _) => s }
      }.toMap
  }
  
  override def toString: String = {
    transitionRewardMap
      .toIndexedSeq
      .sortWith { case ((x, _), (y, _)) => sortingFunction(x, y) }
      .mkString("\n")
  }
  
  def rewardVectorToString: String = {
    rewardVector
      .valuesIterator
      .zip(nonTerminalStates)
      .map { case (r, s) => f"Expected Reward for $s: $r%1.4f" }
      .mkString("\n")
  }
  
  def rewardVector: DenseVector[Double] = {
    val expectedRewards = nonTerminalStates.map { nt =>
      val nextStateRewardDistribution = transitionRewardMap(nt)
      nextStateRewardDistribution.expectation { case (_, r) => r }
    }
    DenseVector[Double](expectedRewards: _*)
  }
  
  def valueFunctionToString(gamma: Double): String = {
    valueFunctionVector(gamma)
      .valuesIterator
      .zip(nonTerminalStates)
      .map { case (v, s) => f"Value for $s: $v%1.4f" }
      .mkString("\n")
  }
  
  def valueFunctionVector(
    gamma: Double // discount factor
  ): DenseVector[Double] = {
    val unitMatrix = DenseMatrix.eye[Double](nonTerminalStates.length)
    val transitionMatrix = this.transitionMatrix
    val A = unitMatrix - gamma * transitionMatrix
    val b = this.rewardVector
    val x = A \ b // solution of linear system Ax = b
    x
  }
}

object FiniteMarkovRewardProcess {
  type StateReward[S] = FiniteDistribution[(State[S], Double)]
  type RewardTransition[S] = Map[NonTerminal[S], StateReward[S]]
  
  def processInputMap[S](
    inputMap: Map[S, FiniteDistribution[(S, Double)]]
  ): RewardTransition[S] = {
    val nonTerminalStates = inputMap.keySet
    
    def toState(stateRewardPair: (S, Double)): (State[S], Double) = {
      val (s, reward) = stateRewardPair
      val state = if (nonTerminalStates.contains(s)) NonTerminal(s) else Terminal(s)
      (state, reward)
    }
    
    inputMap.map { case (s, fd) => NonTerminal(s) -> fd.map(toState) }
  }
  
}