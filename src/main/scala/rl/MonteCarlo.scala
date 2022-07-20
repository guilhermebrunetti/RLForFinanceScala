package rl

import rl.ApproximateDynamicProgramming.{NTStateDistribution, QValueFunctionApproximation, ValueFunctionApproximation}
import rl.utils.{Categorical, Distribution}

/**
 * Monte Carlo methods for working with Markov Reward Process and
 * Markov Decision Processes.
 */
object MonteCarlo {
  
  /**
   * Evaluate an MRP using the monte carlo method, simulating episodes
   * of the given number of steps.
   *
   * Each value this function yields represents the approximated value
   * function for the MRP after one additional episode.
   *
   * Arguments:
   * traces -- an iterator of simulation traces from an MRP
   * initialApproximation -- initial approximation of value function
   * gamma -- discount rate (0 < gamma ≤ 1), default: 1
   * episodeLengthTolerance -- stop iterating once gamma**2 ≤ tolerance
   *
   * Returns an iterator with updates to the approximated value
   * function after each episode.
   *
   */
  def mcPrediction[S](
    traces: Iterable[Iterable[TransitionStep[S]]],
    initialApproximation: ValueFunctionApproximation[S],
    gamma: Double = 1.0,
    episodeLengthTolerance: Double = 1.0e-6
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    val episodes: Iterable[Iterable[ReturnStep[S]]] = traces.map { trace =>
      Returns.returns(trace, gamma, episodeLengthTolerance)
    }
    
    val xySeqStream: Iterable[Iterable[(NonTerminal[S], Double)]] =
      episodes.map { episode => episode.map { step => (step.state, step.returns) } }
    
    IterateUtils.accumulate(xySeqStream.iterator, initialApproximation)(
      (f, xySeq) => IterateUtils.last(f.iterateUpdates(xySeq.map(xy => Seq(xy)))).getOrElse(
        throw new IllegalArgumentException(s"Iterate.iterate method did not converge")
      )
    )
    
  }
  
  def batchMCPrediction[S](
    traces: Iterable[Iterable[TransitionStep[S]]],
    initialApproximation: ValueFunctionApproximation[S],
    gamma: Double = 1.0,
    episodeLengthTolerance: Double = 1.0e-6,
    convergenceTolerance: Double = 1.0e-5
  ): ValueFunctionApproximation[S] = {
    
    val returnSteps: Iterable[ReturnStep[S]] = traces.flatMap { trace =>
      Returns.returns(trace, gamma, episodeLengthTolerance)
    }
    
    val xySeq = returnSteps.map { step => (step.state, step.returns) }
    
    initialApproximation.solve(xySeq, Some(convergenceTolerance))
  }
  
  def batchMCPredictionQValue[S, A](
    traces: Iterable[Iterable[ActionStep[S, A]]],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double = 1.0,
    episodeLengthTolerance: Double = 1.0e-6,
    convergenceTolerance: Double = 1.0e-5
  ): QValueFunctionApproximation[S, A] = {
    
    val returnSteps: Iterable[ReturnStepMDP[S, A]] = traces.flatMap { trace =>
      Returns.returnsMDP(trace, gamma, episodeLengthTolerance)
    }
    
    val xySeq = returnSteps.map { step => ((step.state, step.action), step.returns) }
    
    initialApproximation.solve(xySeq, Some(convergenceTolerance))
  }
  
  /**
   *
   * Evaluate an MDP using the monte carlo method, simulating episodes
   * of the given number of steps.
   *
   * Each value this function yields represents the approximated value
   * function for the MRP after one additional episode.
   *
   * Arguments:
   * markovDecisionProcess -- the Markov Decision Process to evaluate
   * states -- distribution of states to start episodes from
   * initialApproximation -- initial approximation of value function
   * gamma -- discount rate (0 ≤ gamma ≤ 1)
   * epsilonFunction -- a function from the number of episodes
   * to epsilon. epsilon is the fraction of the actions where we explore
   * rather than following the optimal policy
   * episodeLengthTolerance -- stop iterating once gamma**2 ≤ tolerance
   *
   * Returns an iterator with updates to the approximated Q function
   * after each episode.
   */
  def glieMCControl[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double = 1.0,
    epsilonFunction: Int => Double,
    episodeLengthTolerance: Double = 1.0e-6
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    val initialPolicy: Policy[S, A] = epsilonGreedyPolicy(initialApproximation, markovDecisionProcess, 1.0)
    
    IterateUtils.iterate[(QValueFunctionApproximation[S, A], Policy[S, A], Int)](
      step = x => {
        val (qValues, policy, n) = x
        val trace: LazyList[ActionStep[S, A]] = markovDecisionProcess.simulateAction(states, policy)
        val returnSteps: Iterable[ReturnStepMDP[S, A]] = Returns.returnsMDP(trace, gamma, episodeLengthTolerance)
        val xySeq = returnSteps.map { step => ((step.state, step.action), step.returns) }
        val nextQ = xySeq.foldLeft(qValues)((q, xy) => q.update(Seq(xy)))
        val nextEps = epsilonFunction(n + 1)
        val nextPolicy = epsilonGreedyPolicy(nextQ, markovDecisionProcess, nextEps)
        (nextQ, nextPolicy, n + 1)
      },
      initialValue = (initialApproximation, initialPolicy, 0)
    ).map(_._1)
  }
  
  def epsilonGreedyPolicy[S, A](
    qValueFunction: QValueFunctionApproximation[S, A],
    mdp: MarkovDecisionProcess[S, A],
    epsilon: Double = 0.0
  ): Policy[S, A] = {
    
    val uniformPolicy: UniformPolicy[S, A] = (state: S) => mdp.actions(NonTerminal(state))
    val greedyPolicy: DeterministicPolicy[S, A] = greedyPolicyFromQValueFunction(qValueFunction, mdp.actions)
    
    val randomPolicy: RandomPolicy[S, A] = new RandomPolicy[S, A] {
      override def policyChoices: Distribution[Policy[S, A]] = Categorical(
        Map(uniformPolicy -> epsilon, greedyPolicy -> (1.0 - epsilon))
      )
    }
    
    randomPolicy
  }
  
  def greedyPolicyFromQValueFunction[S, A](
    qValueFunction: QValueFunctionApproximation[S, A],
    actions: NonTerminal[S] => Iterable[A]
  ): DeterministicPolicy[S, A] = (state: S) => {
    val nt = NonTerminal(state)
    actions(nt).maxBy(qValueFunction(nt, _))
  }
  
  /**
   *
   * Evaluate an MDP using the monte carlo method, simulating episodes
   * of the given number of steps.
   *
   *
   * Returns an iterator with updates to the approximated Q function
   * after each episode. There is one update per episode.
   */
  def batchMCControl[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double = 1.0,
    epsilonFunction: Int => Double,
    episodeLengthTolerance: Double = 1.0e-6
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    val initialPolicy: Policy[S, A] = epsilonGreedyPolicy(initialApproximation, markovDecisionProcess, 1.0)
    
    IterateUtils.iterate[(QValueFunctionApproximation[S, A], Policy[S, A], Int)](
      step = x => {
        val (q, policy, n) = x
        val trace: LazyList[ActionStep[S, A]] = markovDecisionProcess.simulateAction(states, policy)
        val returnSteps: Iterable[ReturnStepMDP[S, A]] = Returns.returnsMDP(trace, gamma, episodeLengthTolerance)
        val xySeq = returnSteps.map { step => ((step.state, step.action), step.returns) }
        val nextQ = q.update(xySeq)
        val nextEps = epsilonFunction(n + 1)
        val nextPolicy = epsilonGreedyPolicy(nextQ, markovDecisionProcess, nextEps)
        (nextQ, nextPolicy, n + 1)
      },
      initialValue = (initialApproximation, initialPolicy, 0)
    ).map(_._1)
  }
  
}
