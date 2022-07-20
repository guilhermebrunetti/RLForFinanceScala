package rl

import breeze.linalg._
import rl.ApproximateDynamicProgramming._
import rl.FunctionApproximationUtils._
import rl.MonteCarlo.epsilonGreedyPolicy
import rl.TemporalDifference.epsilonGreedyAction
import rl.utils.Categorical
import rl.utils.{Categorical, Distribution}

/**
 * lambda-return and TD(lambda) methods for working with prediction and control
 */
object TemporalDifferenceLambda {
  
  /**
   *
   * Value Function Prediction using the lambda-return method given a
   * sequence of traces.
   *
   * Each value this function yields represents the approximated value
   * function for the MRP after an additional episode
   *
   * Arguments:
   * traces -- a sequence of traces
   * initialApproximation -- initial approximation of value function
   * gamma -- discount rate (0 < gamma ≤ 1)
   * lambda -- lambda parameter (0 <= lambda <= 1)
   */
  def lambdaReturnPrediction[S](
    traces: Iterable[Iterable[TransitionStep[S]]],
    initialApproximation: ValueFunctionApproximation[S],
    gamma: Double,
    lambda: Double
  ): Iterable[ValueFunctionApproximation[S]] = {
    throw new NotImplementedError(s"Method lambdaReturnPrediction is not implemented yet")
  }
  
  /**
   * Evaluate an MRP using TD(lambda) using the given sequence of traces.
   *
   * Each value this function yields represents the approximated value function
   * for the MRP after an additional transition within each trace
   *
   * Arguments:
   * traces -- a sequence of transitions from an MRP which don't
   * have to be in order or from the same simulation
   * initialApproximation -- initial approximation of value function
   * gamma -- discount rate (0 < gamma ≤ 1)
   * lambda -- lambda parameter (0 <= lambda <= 1)
   *
   */
  def tdLambdaPrediction[S](
    traces: Iterable[Iterable[TransitionStep[S]]],
    initialApproximation: FunctionApproximation[NonTerminal[S]],
    gamma: Double,
    lambda: Double
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    val flattenedTraces: Iterable[(TransitionStep[S], Int)] = traces.zipWithIndex.flatMap {
      case (trace, i) => trace.map(step => (step, i))
    }
  
    type X = FunctionApproximation[NonTerminal[S]]
    type Y = Gradient[NonTerminal[S], FunctionApproximation]
    
    val eligibilityTrace: Y = gradientZero(initialApproximation)
    
    IterateUtils.accumulate(
      flattenedTraces.iterator,
      (initialApproximation, eligibilityTrace, 0)
    ) { case ((f, previousETr, i), (step, j)) =>
      val eTr = if (i == j) previousETr else gradientZero(f)
      val x: NonTerminal[S] = step.state
      val y: Double = step.reward + gamma * extendedValueFunction(f)(step.nextState)
      val xySeq: Seq[(NonTerminal[S], Double)] = Seq((x, y))
      val nextETr: Y = updateEligibilityTrace(f, eTr, gamma * lambda, xySeq)
      val nextF: X = updateWithEligibilityTrace(f, nextETr, x, y)
      (nextF, nextETr, j)
    }.map(_._1)
  }
  
  def glieSarsaLambda[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: FunctionApproximation[(NonTerminal[S], A)],
    gamma: Double = 1.0,
    lambda: Double,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    type X = FunctionApproximation[(NonTerminal[S], A)]
    type Y = Gradient[(NonTerminal[S], A), FunctionApproximation]
    
    val initialState: NonTerminal[S] = initialStateDistribution.sample
    val initialAction: Option[A] = None
    val eligibilityTrace: Y = gradientZero(initialApproximation)
    
    IterateUtils.iterate[(X, Y, Int, Int, NonTerminal[S], Option[A])](
      step = x => {
        val (q, eTr, episode, step, state, actionOpt) = x
        val epsilon: Double = epsilonFunction(episode)
        val action = actionOpt.getOrElse(epsilonGreedyAction(q, state, markovDecisionProcess.actions(state), epsilon))
        val stateActionPair = (state, action)
        val (nextState: State[S], reward: Double) = markovDecisionProcess.step(state, action).sample
        nextState match {
          case nt: NonTerminal[S] if step < maxEpisodeLength =>
            val nextAction = epsilonGreedyAction(q, nt, markovDecisionProcess.actions(nt), epsilon)
            val y = reward + gamma * q(nt, nextAction)
            val xySeq = Seq((stateActionPair, y))
            val nextETr: Y = updateEligibilityTrace(q, eTr, gamma * lambda, xySeq)
            val nextQ: X = updateWithEligibilityTrace(q, nextETr, stateActionPair, y)
            (nextQ, nextETr, episode, step + 1, nt, Some(nextAction))
          case _ =>
            val nextQ = q.update(Seq((stateActionPair, reward)))
            val newState = initialStateDistribution.sample
            val nextEtr = gradientZero(q)
            (nextQ, nextEtr, episode + 1, 0, newState, None)
        }
      },
      initialValue = (initialApproximation, eligibilityTrace, 1, 0, initialState, initialAction)
    ).map(_._1)
  }
  
  
  def tdLambdaBatchPrediction[S](
    traces: Iterable[Iterable[TransitionStep[S]]],
    initialApproximation: ValueFunctionApproximation[S],
    gamma: Double,
    lambda: Double
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    IterateUtils.accumulate(
      traces.iterator,
      initialApproximation
    ) {
      (valueFunction, trace) => tdLambdaStep(valueFunction, trace, gamma, lambda)
    }
    
  }
  
  private def tdLambdaStep[S](
    valueFunction: FunctionApproximation[NonTerminal[S]],
    trace: Iterable[TransitionStep[S]],
    gamma: Double,
    lambda: Double
  ): FunctionApproximation[NonTerminal[S]] = {
    val eligibilityTrace: Gradient[NonTerminal[S], FunctionApproximation] = gradientZero(valueFunction)
    val (finalValueFunction, _) = trace.foldLeft((valueFunction, eligibilityTrace)) { case ((f, eTr), step) =>
      val x: NonTerminal[S] = step.state
      val y: Double = step.reward + gamma * extendedValueFunction(f)(step.nextState)
      val xySeq: Seq[(NonTerminal[S], Double)] = Seq((x, y))
      val nextETr: Gradient[NonTerminal[S], FunctionApproximation] = updateEligibilityTrace(f, eTr, gamma * lambda, xySeq)
      val nextF: FunctionApproximation[NonTerminal[S]] = updateWithEligibilityTrace(f, nextETr, x, y)
      (nextF, nextETr)
    }
    finalValueFunction
  }
  
  
}
