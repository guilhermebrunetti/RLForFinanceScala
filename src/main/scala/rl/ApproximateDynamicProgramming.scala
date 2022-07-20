package rl

import breeze.linalg._
import rl.FiniteMarkovRewardProcess.RewardTransition
import rl.IterateUtils
import rl.utils.Distribution

/**
 * Approximate dynamic programming algorithms are variations on
 * dynamic programming algorithms that can work with function
 * approximations rather than exact representations of the process's
 * state space.
 */
object ApproximateDynamicProgramming {
  
  implicit val defaultTolerance: Double = 1.0e-5
  
  // A representation of a value function for a finite MDP
  // with states of type S
  type ValueFunctionApproximation[S] = FunctionApproximation[NonTerminal[S]]
  type QValueFunctionApproximation[S, A] = FunctionApproximation[(NonTerminal[S], A)]
  type NTStateDistribution[S] = Distribution[NonTerminal[S]]
  type MRPValueFunctionApproxDistribution[S] =
    (MarkovRewardProcess[S], ValueFunctionApproximation[S], NTStateDistribution[S])
  type MDPValueFuncApproxDistribution[S, A] =
    (MarkovDecisionProcess[S, A], ValueFunctionApproximation[S], NTStateDistribution[S])
  type MDPQValueFuncApproxDistribution[S, A] =
    (MarkovDecisionProcess[S, A], QValueFunctionApproximation[S, A], NTStateDistribution[S])
  
  def almostEqual[X](
    x: FunctionApproximation[X],
    y: FunctionApproximation[X]
  )(implicit tolerance: Double = defaultTolerance): Boolean = {
    
    (x, y) match {
      case (a: Dynamic[X], b: Dynamic[X]) => a.within(b, tolerance)
      case (a: Tabular[X], b: Tabular[X]) => a.within(b, tolerance)
      case (a: LinearFunctionApproximation[X], b: LinearFunctionApproximation[X]) => a.within(b, tolerance)
      case (a: DNNApproximation[X], b: DNNApproximation[X]) => a.within(b, tolerance)
      case _ => throw new IllegalArgumentException(s"Incompatible classes between x (${x.getClass}) and y (${y.getClass})")
    }
  }
  
  def evaluateFiniteMRP[S](
    mrp: FiniteMarkovRewardProcess[S],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S]
  ): Iterator[ValueFunctionApproximation[S]] = {
    def update(v: ValueFunctionApproximation[S]): ValueFunctionApproximation[S] = {
      val ys = v.evaluate(mrp.nonTerminalStates)
      val updated = mrp.rewardVector + gamma * mrp.transitionMatrix * ys
      v.update(mrp.nonTerminalStates.zip(updated.valuesIterator))
    }
    
    IterateUtils.iterate(update, initialApproximation)
  }
  
  def evaluateFiniteMRPResult[S](
    mrp: FiniteMarkovRewardProcess[S],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S]
  )(implicit tolerance: Double = defaultTolerance): ValueFunctionApproximation[S] = {
   IterateUtils.converged[ValueFunctionApproximation[S]](
     evaluateFiniteMRP(mrp, gamma, initialApproximation),
     done = almostEqual(_, _)(tolerance)
   )
  }
  
  /**
   * Iteratively calculate the value function for the give finite Markov
   * Reward Process, using the given FunctionApprox to approximate the
   * value function at each step.
   */
  def evaluateMarkovRewardProcess[S](
    mrp: FiniteMarkovRewardProcess[S],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S],
    ntStateDistribution: NTStateDistribution[S],
    numSamples: Int
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    def update(v: ValueFunctionApproximation[S]): ValueFunctionApproximation[S] = {
      val ntStates: Seq[NonTerminal[S]] = ntStateDistribution.samples(numSamples)
      
      def mrpReturn(stateReward: (State[S], Double)): Double = {
        val (state, reward) = stateReward
        reward + gamma * extendedValueFunction(v)(state)
      }
      
      val updated = ntStates.map { nt =>
        nt -> mrp.transitionReward(nt).expectation(mrpReturn)
      }
      
      v.update(updated)
    }
    
    IterateUtils.iterate(update, initialApproximation)
  }
  
  def evaluateMarkovRewardProcessResult[S](
    mrp: FiniteMarkovRewardProcess[S],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S],
    ntStateDistribution: NTStateDistribution[S],
    numSamples: Int
  )(implicit tolerance: Double = defaultTolerance): ValueFunctionApproximation[S] = {
    IterateUtils.converged[ValueFunctionApproximation[S]](
      evaluateMarkovRewardProcess(mrp, gamma, initialApproximation, ntStateDistribution, numSamples),
      done = almostEqual(_, _)(tolerance)
    )
  }
  
  def extendedValueFunction[S](valueFunction: ValueFunctionApproximation[S])(state: State[S]): Double = {
    state.onNonTerminal(valueFunction.apply, 0.0)
  }
  
  /**
   * Iteratively calculate the Optimal Value function for the given finite
   * Markov Decision Process, using the given FunctionApproximation to approximate the
   * Optimal Value function at each step
   */
  def valueIterationFinite[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S]
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    def update(v: ValueFunctionApproximation[S]): ValueFunctionApproximation[S] = {
      
      def mrpReturn(stateReward: (State[S], Double)): Double = {
        val (state, reward) = stateReward
        reward + gamma * extendedValueFunction(v)(state)
      }
      
      val updated = mdp.nonTerminalStates.map { nt =>
        nt -> mdp.actions(nt).map { action =>
          mdp.stateActionMap(nt)(action).expectation(mrpReturn)
        }.max
      }
      
      v.update(updated)
    }
    
    IterateUtils.iterate(update, initialApproximation)
  }
  
  def valueIterationFiniteResult[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S]
  )(implicit tolerance: Double = defaultTolerance): ValueFunctionApproximation[S] = {
    IterateUtils.converged[ValueFunctionApproximation[S]](
      valueIterationFinite(mdp, gamma, initialApproximation),
      done = almostEqual(_, _)(tolerance)
    )
  }
  
  /**
   * Iteratively calculate the Optimal Value function for the given
   * Markov Decision Process, using the given FunctionApproximation to approximate the
   * Optimal Value function at each step for a random sample of the process'
   * non-terminal states.
   */
  def valueIteration[S, A](
    mdp: MarkovDecisionProcess[S, A],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S],
    ntStateDistribution: NTStateDistribution[S],
    numSamples: Int
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    def update(v: ValueFunctionApproximation[S]): ValueFunctionApproximation[S] = {
      val ntStates: Seq[NonTerminal[S]] = ntStateDistribution.samples(numSamples)
      
      def mrpReturn(stateReward: (State[S], Double)): Double = {
        val (state, reward) = stateReward
        reward + gamma * extendedValueFunction(v)(state)
      }
      
      val updated = ntStates.map { nt =>
        nt -> mdp.actions(nt).map { action =>
          mdp.step(nt, action).expectation(mrpReturn)
        }.max
      }
      
      v.update(updated)
    }
    
    IterateUtils.iterate(update, initialApproximation)
  }
  
  def valueIterationResult[S, A](
    mdp: MarkovDecisionProcess[S, A],
    gamma: Double,
    initialApproximation: ValueFunctionApproximation[S],
    ntStateDistribution: NTStateDistribution[S],
    numSamples: Int
  )(implicit tolerance: Double = defaultTolerance): ValueFunctionApproximation[S] = {
    IterateUtils.converged[ValueFunctionApproximation[S]](
      valueIteration(mdp, gamma, initialApproximation, ntStateDistribution, numSamples),
      done = almostEqual(_, _)(tolerance)
    )
  }
  
  /**
   * Evaluate the given finite Markov Reward Process using backwards
   * induction, given that the process stops after limit time steps.
   */
  def backwardInductionFinite[S](
    stepFunctionSeq: Seq[(RewardTransition[S], ValueFunctionApproximation[S])],
    gamma: Double
  ): Seq[ValueFunctionApproximation[S]] = {
    
    val vfOption: Option[ValueFunctionApproximation[S]] = None
    
    val valueFunctions: Seq[Option[ValueFunctionApproximation[S]]] =
      stepFunctionSeq.scanRight(vfOption) { case ((step, approx), vOpt) =>
        def mrpReturn(stateReward: (State[S], Double)): Double = {
          val (state, reward) = stateReward
          reward + gamma * vOpt.map(extendedValueFunction(_)(state)).getOrElse(0.0)
        }
        
        val target: Map[NonTerminal[S], Double] = step.map { case (state, rewardTransition) =>
          state -> rewardTransition.expectation(mrpReturn)
        }
        Some(approx.solve(target))
      }
    
    valueFunctions.init.flatten
  }
  
  /**
   * Evaluate the given finite Markov Reward Process using backwards
   * induction, given that the process stops after limit time steps, using
   * the given FunctionApproximation for each time step for a random sample of the
   * time step's states.
   */
  def backwardInduction[S](
    mrpFunctionDistribution: Seq[MRPValueFunctionApproxDistribution[S]],
    gamma: Double,
    numSamples: Int,
    errorTolerance: Double
  ): Seq[ValueFunctionApproximation[S]] = {
    
    val vfOption: Option[ValueFunctionApproximation[S]] = None
    
    val valueFunctions: Seq[Option[ValueFunctionApproximation[S]]] =
      mrpFunctionDistribution.scanRight(vfOption) { case ((mrp, approx, mu), vOpt) =>
        def mrpReturn(stateReward: (State[S], Double)): Double = {
          val (state, reward) = stateReward
          reward + gamma * vOpt.map(extendedValueFunction(_)(state)).getOrElse(0.0)
        }
        
        val target: Seq[(NonTerminal[S], Double)] =
          mu.samples(numSamples).map(state =>
            state -> mrp.transitionReward(state).expectation(mrpReturn))
        Some(approx.solve(target, Some(errorTolerance)))
      }
    
    valueFunctions.init.flatten
  }
  
  def backwardOptimalValueFunctionAndPolicy[S, A](
    mdpFunctionDistribution: Seq[MDPValueFuncApproxDistribution[S, A]],
    gamma: Double,
    numSamples: Int,
    errorTolerance: Double
  ): Seq[(ValueFunctionApproximation[S], DeterministicPolicy[S, A])] = {
    
    val vfPolicyOption: Option[(ValueFunctionApproximation[S], DeterministicPolicy[S, A])] = None
    
    val valueFunctionsAndPolicies: Seq[Option[(ValueFunctionApproximation[S], DeterministicPolicy[S, A])]] =
      mdpFunctionDistribution.scanRight(vfPolicyOption) { case ((mdp, approx, mu), vPiOpt) =>
        def mrpReturn(stateReward: (State[S], Double)): Double = {
          val (state, reward) = stateReward
          reward + gamma * vPiOpt.map { case (vf, _) => extendedValueFunction(vf)(state) }.getOrElse(0.0)
        }
        
        val target: Seq[(NonTerminal[S], Double)] =
          mu.samples(numSamples).map(state =>
            state -> mdp.actions(state)
              .map(action => mdp.step(state, action).expectation(mrpReturn))
              .max
          )
        
        val valueFunction = approx.solve(target, Some(errorTolerance))
        
        val deterministicPolicy: DeterministicPolicy[S, A] = (state: S) => {
          val nt = NonTerminal(state)
          mdp.actions(nt)
            .maxBy { action => mdp.step(nt, action).expectation(mrpReturn) }
        }
        
        Some((valueFunction, deterministicPolicy))
      }
    
    valueFunctionsAndPolicies.init.flatten
  }
  
  def backwardOptimalQValueFunction[S, A](
    mdpQFunctionDistribution: Seq[MDPQValueFuncApproxDistribution[S, A]],
    gamma: Double,
    numSamples: Int,
    errorTolerance: Double
  ): Seq[QValueFunctionApproximation[S, A]] = {
    
    val qVFOption: Option[QValueFunctionApproximation[S, A]] = None
    
    val qValueFunctions =
      mdpQFunctionDistribution.scanRight(qVFOption) { case ((mdp, approx, mu), qvfOpt) =>
        def mrpReturn(stateReward: (State[S], Double)): Double = {
          val (state, reward) = stateReward
          val vf = state.onNonTerminal(
            f = nt => qvfOpt.map { qvf =>
              mdp.actions(nt).map { action => qvf(nt, action) }.max
            }.getOrElse(0.0),
            defaultValue = 0.0)
          reward + gamma * vf
        }
        
        val target: Seq[((NonTerminal[S], A), Double)] =
          mu.samples(numSamples).flatMap { state =>
            mdp.actions(state).map { action =>
              (state, action) -> mdp.step(state, action).expectation(mrpReturn)
            }
          }
        
        val qValueFunction = approx.solve(target, Some(errorTolerance))
        
        Some(qValueFunction)
      }
    
    qValueFunctions.init.flatten
  }
  
}
