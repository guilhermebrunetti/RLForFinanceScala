package rl

import breeze.numerics._

object Returns {
  
  def returnsMDP[S, A](
    trace: Iterable[ActionStep[S, A]],
    gamma: Double = 1.0,
    tolerance: Double = 1.0e-6
  ): Iterable[ReturnStepMDP[S, A]] = {
    val returnsMRP = returns(trace, gamma, tolerance)
    returnsMRP.collect {
      case ret: ReturnStepMDP[S, A]@unchecked => ret
    }
  }
  
  /**
   *
   * Given an iterator of states and rewards, calculate the return of
   * the first N states.
   *
   * Arguments:
   * rewards -- instantaneous rewards
   * gamma -- the discount factor (0 < gamma ≤ 1)
   * tolerance -- a small value—we stop iterating once pow(gamma, steps) ≤ tolerance
   */
  def returns[S](
    trace: Iterable[TransitionStep[S]],
    gamma: Double = 1.0,
    tolerance: Double = 1.0e-6
  ): Iterable[ReturnStep[S]] = {
    
    val maxStepsOpt: Option[Int] = if (gamma < 1) Some(round(log(tolerance) / log(gamma)).toInt) else None
    // For non-episodic tasks, we simulate traces of length 2 * maxSteps, but then we discard the last maxSteps
    val tr = maxStepsOpt.map(maxSteps => trace.take(2 * maxSteps)).getOrElse(trace)
    val initialTrace = tr.init
    val lastValue: TransitionStep[S] = tr.last
    
    def accumulator(
      next: TransitionStep[S],
      curr: ReturnStep[S]
    ): ReturnStep[S] = next.addReturn(curr.returns, gamma)
    
    val returns = IterateUtils.accumulateRight[TransitionStep[S], ReturnStep[S]](
      values = initialTrace,
      initialValue = lastValue.addReturn(0.0, gamma))(
      f = accumulator
    )
    
    maxStepsOpt.map(maxSteps => returns.take(maxSteps)).getOrElse(returns)
  }
  
}
