package rl

import breeze.numerics._
import breeze.linalg._
import rl.ApproximateDynamicProgramming._
import rl.utils.{Categorical, Distribution}

/**
 * Temporal Difference Monte Carlo methods for working with Markov Reward Process and
 * Markov Decision Processes.
 */
object TemporalDifference {
  
  type PolicyFromQValue[S, A] = (QValueFunctionApproximation[S, A], MarkovDecisionProcess[S, A]) => Policy[S, A]
  
  def tdPrediction[S](
    transitions: Iterable[TransitionStep[S]],
    initialApproximation: ValueFunctionApproximation[S],
    gamma: Double = 1.0,
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    def step(
      functionApproximation: ValueFunctionApproximation[S],
      transition: TransitionStep[S],
    ): ValueFunctionApproximation[S] = {
      val x = transition.state
      val y = transition.reward + gamma * extendedValueFunction(functionApproximation)(transition.nextState)
      functionApproximation.update(Seq((x, y)))
    }
    
    IterateUtils.accumulate(transitions.iterator, initialApproximation)(step)
  }
  
  def batchTDPrediction[S](
    transitions: Iterable[TransitionStep[S]],
    initialApproximation: ValueFunctionApproximation[S],
    gamma: Double = 1.0,
    convergenceTolerance: Double = 1.0e-5
  ): ValueFunctionApproximation[S] = {
    
    def step(
      functionApproximation: ValueFunctionApproximation[S],
      transitions: Iterable[TransitionStep[S]]
    ): ValueFunctionApproximation[S] = {
      val xySeq = transitions.map { transition =>
        val x = transition.state
        val y = transition.reward + gamma * extendedValueFunction(functionApproximation)(transition.nextState)
        (x, y)
      }
      functionApproximation.update(xySeq)
    }
    
    def done(
      x: ValueFunctionApproximation[S],
      y: ValueFunctionApproximation[S],
    ): Boolean = {
      almostEqual(x, y)(convergenceTolerance)
    }
    
    IterateUtils.converged(
      values = IterateUtils.accumulate(Iterator.continually(transitions), initialApproximation)(step),
      done = done
    )
  }
  
  def leastSquaresTD[S](
    transitions: Iterable[TransitionStep[S]],
    featureFunctions: Seq[NonTerminal[S] => Double],
    gamma: Double,
    epsilon: Double
  ): LinearFunctionApproximation[NonTerminal[S]] = {
    val numFeatures = featureFunctions.size
    val a0: DenseMatrix[Double] = DenseMatrix.eye[Double](numFeatures) / epsilon
    val b0: DenseVector[Double] = DenseVector.zeros[Double](numFeatures)
    val (a, b) = transitions.foldLeft((a0, b0)) { case ((mat, vec), transitionStep) =>
      val featureValues = featureFunctions.map { f => f(transitionStep.state) }
      val phi1 = DenseVector.apply(featureValues.toArray)
      val phi2 = transitionStep.nextState.onNonTerminal(
        f = nt => phi1 - gamma * DenseVector(featureFunctions.map { f => f(nt) }.toArray),
        defaultValue = phi1
      )
      val temp: DenseVector[Double] = mat.t * phi2
      val newMat: DenseMatrix[Double] = mat - ((mat * phi1) * temp.t) / (1 + phi1.t * temp)
      val newVec: DenseVector[Double] = vec + phi1 * transitionStep.reward
      (newMat, newVec)
    }
    val optWeights = a * b
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      weightsOption = Some(Weights.create(weights = optWeights))
    )
  }
  
  def glieSarsa[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double = 1.0,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    val initialState: NonTerminal[S] = initialStateDistribution.sample
    
    IterateUtils.iterate[(QValueFunctionApproximation[S, A], Int, Int, NonTerminal[S], Option[A])](
      step = x => {
        val (q, episode, step, state, actionOpt) = x
        val epsilon: Double = epsilonFunction(episode)
        val action = actionOpt.getOrElse(epsilonGreedyAction(q, state, markovDecisionProcess.actions(state), epsilon))
        val stateActionPair = (state, action)
        val (nextState: State[S], reward: Double) = markovDecisionProcess.step(state, action).sample
        nextState match {
          case nt: NonTerminal[S] if step < maxEpisodeLength =>
            val nextAction = epsilonGreedyAction(q, nt, markovDecisionProcess.actions(nt), epsilon)
            val nextQ = q.update(Seq((stateActionPair, reward + gamma * q(nt, nextAction))))
            (nextQ, episode, step + 1, nt, Some(nextAction))
          case _ =>
            val nextQ = q.update(Seq((stateActionPair, reward)))
            val newState = initialStateDistribution.sample
            (nextQ, episode + 1, 0, newState, None)
        }
      },
      initialValue = (initialApproximation, 1, 0, initialState, None)
    ).map(_._1)
  }
  
  def epsilonGreedyAction[S, A](
    qValueFunction: QValueFunctionApproximation[S, A],
    nonTerminal: NonTerminal[S],
    actions: Iterable[A],
    epsilon: Double
  ): A = {
    val uniformPolicy: UniformPolicy[S, A] = (_: S) => actions
    val greedyPolicy: DeterministicPolicy[S, A] = (state: S) => actions.maxBy(qValueFunction(NonTerminal(state), _))
    
    val randomPolicy: RandomPolicy[S, A] = new RandomPolicy[S, A] {
      override def policyChoices: Distribution[Policy[S, A]] = Categorical(
        Map(uniformPolicy -> epsilon, greedyPolicy -> (1.0 - epsilon))
      )
    }
    randomPolicy.act(nonTerminal).sample
  }
  
  def qLearning[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    policyFromQValue: PolicyFromQValue[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    val initialState: NonTerminal[S] = initialStateDistribution.sample
    
    IterateUtils.iterate[(QValueFunctionApproximation[S, A], Int, Int, NonTerminal[S])](
      step = x => {
        val (q, episode, step, state) = x
        val policy = policyFromQValue(q, markovDecisionProcess)
        val action = policy.act(state).sample
        val stateActionPair = (state, action)
        val (nextState: State[S], reward: Double) = markovDecisionProcess.step(state, action).sample
        nextState match {
          case nt: NonTerminal[S] if step < maxEpisodeLength =>
            val nextReturn = markovDecisionProcess.actions(nt).map(q(nt, _)).max
            val nextQ = q.update(Seq((stateActionPair, reward + gamma * nextReturn)))
            (nextQ, episode, step + 1, nt)
          case _ =>
            val nextQ = q.update(Seq((stateActionPair, reward)))
            val newState = initialStateDistribution.sample
            (nextQ, episode + 1, 0, newState)
        }
      },
      initialValue = (initialApproximation, 1, 0, initialState)
    ).map(_._1)
  }
  
  /**
   *
   * Return policies that try to maximize the reward based on the given
   * set of experiences.
   *
   * Arguments:
   * transitions -- a sequence of state, action, reward, state (S, A, R, S')
   * actions -- a function returning the possible actions for a given state
   * initialApproximation -- initial approximation of q function
   * gamma -- discount rate (0 < gamma ≤ 1)
   *
   * Returns:
   * an iterator of approximations of the q function based on the
   * transitions given as input
   */
  def qLearningExternalTransitions[S, A](
    transitions: Iterable[ActionStep[S, A]],
    actions: NonTerminal[S] => Iterable[A],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double,
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    def step(
      q: QValueFunctionApproximation[S, A],
      transition: ActionStep[S, A]
    ): QValueFunctionApproximation[S, A] = {
      val nextReturn = transition.nextState.onNonTerminal(
        f = nt => actions(nt).map(q(nt, _)).max,
        defaultValue = 0.0
      )
      val x = (transition.state, transition.action)
      val y = transition.reward + gamma * nextReturn
      q.update(Seq((x, y)))
    }
    
    IterateUtils.accumulate(transitions.iterator, initialApproximation)(step)
  }
  
  def qLearningExperienceReplay[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    policyFromQValue: PolicyFromQValue[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double,
    maxEpisodeLength: Int,
    miniBatchSize: Int,
    weightsDecayHalfLife: Double
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    val experienceReplayMemory = ExperienceReplayMemory[ActionStep[S, A]](
      timeWeightsFunction = (t: Int) => pow(0.5, t.toDouble / weightsDecayHalfLife)
    )
    
    val initialState: NonTerminal[S] = initialStateDistribution.sample
    
    IterateUtils.iterate[(QValueFunctionApproximation[S, A], ExperienceReplayMemory[ActionStep[S, A]], Int, NonTerminal[S])](
      step = x => {
        val (q, expReplay, step, state) = x
        val policy = policyFromQValue(q, markovDecisionProcess)
        val action = policy.act(state).sample
        val (nextState: State[S], reward: Double) = markovDecisionProcess.step(state, action).sample
        val nextExpReplay = expReplay.addData(ActionStep(state, action, nextState, reward))
        val transitions = nextExpReplay.sampleMiniBatch(miniBatchSize)
        val xySeq = transitions.map { step =>
          val nextReturn = step
            .nextState
            .onNonTerminal(
              f = (nt: NonTerminal[S]) => markovDecisionProcess.actions(nt).map(a => q((nt, a))).max,
              defaultValue = 0.0
            )
          ((step.state, step.action), step.reward + gamma * nextReturn)
        }
        val nextQ = q.update(xySeq)
        nextState match {
          case nt: NonTerminal[S] if step < maxEpisodeLength =>
            (nextQ, nextExpReplay, step + 1, nt)
          case _ =>
            val newState = initialStateDistribution.sample
            (nextQ, nextExpReplay, 0, newState)
        }
      },
      initialValue = (initialApproximation, experienceReplayMemory, 0, initialState)
    ).map(_._1)
  }
  
  def leastSquaresTDQ[S, A](
    transitions: Iterable[ActionStep[S, A]], // Finite iterable
    featureFunctions: Seq[((NonTerminal[S], A)) => Double],
    targetPolicy: DeterministicPolicy[S, A],
    gamma: Double,
    epsilon: Double
  ): LinearFunctionApproximation[(NonTerminal[S], A)] = {
    val numFeatures = featureFunctions.size
    
    val a0: DenseMatrix[Double] = DenseMatrix.eye[Double](numFeatures) / epsilon
    val b0: DenseVector[Double] = DenseVector.zeros[Double](numFeatures)
    val (a, b) = transitions.foldLeft((a0, b0)) { case ((mat, vec), transitionStep) =>
      val featureValues: Seq[Double] = featureFunctions.map { f => f(transitionStep.state, transitionStep.action) }
      val phi1 = DenseVector.apply(featureValues.toArray)
      val phi2 = transitionStep.nextState.onNonTerminal(
        f = nt => {
          val targetVector = featureFunctions.map(q => q(nt, targetPolicy.actionForState(nt.state)))
          phi1 - gamma * DenseVector(targetVector.toArray)
        },
        defaultValue = phi1
      )
      val temp: DenseVector[Double] = mat.t * phi2
      val newMat: DenseMatrix[Double] = mat - ((mat * phi1) * temp.t) / (1 + phi1.t * temp)
      val newVec: DenseVector[Double] = vec + phi1 * transitionStep.reward
      (newMat, newVec)
    }
    val optWeights = a * b
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      weightsOption = Some(Weights.create(weights = optWeights))
    )
  }
  
  def leastSquaresPolicyIteration[S, A](
    transitions: Iterable[ActionStep[S, A]],
    actions: NonTerminal[S] => Iterable[A],
    featureFunctions: Seq[((NonTerminal[S], A)) => Double],
    initialTargetPolicy: DeterministicPolicy[S, A],
    gamma: Double,
    epsilon: Double
  ): Iterator[LinearFunctionApproximation[(NonTerminal[S], A)]] = {
    
    Iterator.unfold(initialTargetPolicy){ policy =>
      val qValueFunction = leastSquaresTDQ(
        transitions = transitions,
        featureFunctions = featureFunctions,
        targetPolicy = policy,
        gamma = gamma,
        epsilon = epsilon
      )
      val nextPolicy = MonteCarlo.greedyPolicyFromQValueFunction(qValueFunction, actions)
      Some((qValueFunction, nextPolicy))
    }
  }
  
}
