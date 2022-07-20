package rl.utils

import breeze.numerics._
import breeze.linalg._
import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming.{NTStateDistribution, QValueFunctionApproximation}
import rl.DynamicProgramming.{ValueFunction, valueIterationResult}
import rl.Tabular.learningRateSchedule
import rl.TemporalDifference.PolicyFromQValue
import rl._

object ControlUtils {
  
  val logger: Logger = Logger("ControlUtils")
  
  def glieMCFiniteControlWeights[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    epsilonFunction: Int => Double,
    episodeLengthTolerance: Double = 1.0e-5
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    
    MonteCarlo.glieMCControl(
      markovDecisionProcess = finiteMarkovDecisionProcess,
      states = Choose(finiteMarkovDecisionProcess.nonTerminalStates),
      initialApproximation = Tabular[(NonTerminal[S], A)](),
      gamma = gamma,
      epsilonFunction = epsilonFunction,
      episodeLengthTolerance = episodeLengthTolerance
    )
  }
  
  def glieMCControlLearningRate[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double,
    epsilonFunction: Int => Double,
    episodeLengthTolerance: Double = 1.0e-5
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    MonteCarlo.glieMCControl(
      markovDecisionProcess = markovDecisionProcess,
      states = initialStateDistribution,
      initialApproximation = initialApproximation,
      gamma = gamma,
      epsilonFunction = epsilonFunction,
      episodeLengthTolerance = episodeLengthTolerance
    )
  }
  
  def glieMCFiniteLearningRateCorrectness[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    epsilonFunction: Int => Double,
    episodeLengthTolerance: Double = 1.0e-5,
    numEpisodes: Int
  ): Unit = {
    
    val qValueFunctions = glieMCFiniteControlLearningRate(
        finiteMarkovDecisionProcess = finiteMarkovDecisionProcess,
        gamma = gamma,
        initialLearningRate = initialLearningRate,
        halfLife = halfLife,
        exponent = exponent,
        epsilonFunction = epsilonFunction,
        episodeLengthTolerance = episodeLengthTolerance
      )
    
    val finalQValueFunction: QValueFunctionApproximation[S, A] = qValueFunctions.drop(numEpisodes).next()
    val (valueFunction, policy) =
      getValueFunctionAndPolicyFromQValueFunction(finiteMarkovDecisionProcess, finalQValueFunction)
    
    val (trueValueFunction, truePolicy) = valueIterationResult(finiteMarkovDecisionProcess, gamma)
    
    val nonTerminalStates = finiteMarkovDecisionProcess.nonTerminalStates
    val valueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${valueFunction(s)}%1.4f" }
      .mkString("\n")
    
    val trueValueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${trueValueFunction(s)}%1.4f" }
      .mkString("\n")
    
    val trueValues = DenseVector.apply(nonTerminalStates.toArray).map(trueValueFunction.apply)
    val pred = DenseVector.apply(nonTerminalStates.toArray).map(valueFunction.apply)
    val error = pred - trueValues
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"GLIE MC Optimal Value Function with $numEpisodes episodes:\n$valueFunctionStr")
    logger.info(f"GLIE MC Optimal Policy with $numEpisodes episodes:\n$policy")
    logger.info(f"True Optimal Value Function:\n$trueValueFunctionStr")
    logger.info(f"True Optimal Policy:\n$truePolicy")
    logger.info(f"RMSE: $rmse%1.4f")
    logger.info(f"Counts Map:\n${finalQValueFunction.asInstanceOf[Tabular[(NonTerminal[S], A)]].countsMapToString}")
  }
  
  def glieMCFiniteControlLearningRate[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    epsilonFunction: Int => Double,
    episodeLengthTolerance: Double = 1.0e-5
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)(_)
    
    MonteCarlo.glieMCControl(
      markovDecisionProcess = finiteMarkovDecisionProcess,
      states = Choose(finiteMarkovDecisionProcess.nonTerminalStates),
      initialApproximation = Tabular[(NonTerminal[S], A)](countToWeight = learningRateFunction),
      gamma = gamma,
      epsilonFunction = epsilonFunction,
      episodeLengthTolerance = episodeLengthTolerance
    )
  }
  
  def glieSarsaFiniteLearningRateCorrectness[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int,
    numUpdates: Int
  ): Unit = {
    
    val qValueFunctions = glieSarsaFiniteLearningRate(
      finiteMarkovDecisionProcess = finiteMarkovDecisionProcess,
      gamma = gamma,
      initialLearningRate = initialLearningRate,
      halfLife = halfLife,
      exponent = exponent,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = maxEpisodeLength
    )
    
    val finalQValueFunction: QValueFunctionApproximation[S, A] = qValueFunctions.drop(numUpdates).next()
    val (valueFunction, policy) =
      getValueFunctionAndPolicyFromQValueFunction(finiteMarkovDecisionProcess, finalQValueFunction)
    
    val (trueValueFunction, truePolicy) = valueIterationResult(finiteMarkovDecisionProcess, gamma)
    
    val nonTerminalStates = finiteMarkovDecisionProcess.nonTerminalStates
    val valueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${valueFunction(s)}%1.4f" }
      .mkString("\n")
    
    val trueValueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${trueValueFunction(s)}%1.4f" }
      .mkString("\n")
    
    val trueValues = DenseVector.apply(nonTerminalStates.toArray).map(trueValueFunction.apply)
    val pred = DenseVector.apply(nonTerminalStates.toArray).map(valueFunction.apply)
    val error = pred - trueValues
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"GLIE SARSA (Tabular) Optimal Value Function with $numUpdates updates of length $maxEpisodeLength:\n$valueFunctionStr")
    logger.info(f"GLIE SARSA (Tabular) Optimal Policy with $numUpdates updates of length $maxEpisodeLength:\n$policy")
    logger.info(f"True Optimal Value Function:\n$trueValueFunctionStr")
    logger.info(f"True Optimal Policy:\n$truePolicy")
    logger.info(f"RMSE: $rmse%1.4f")
    logger.info(f"Counts Map:\n${finalQValueFunction.asInstanceOf[Tabular[(NonTerminal[S], A)]].countsMapToString}")
  }
  
  def getValueFunctionAndPolicyFromQValueFunction[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    qValueFunction: QValueFunctionApproximation[S, A]
  ): (ValueFunction[S], FiniteDeterministicPolicy[S, A]) = {
    val nonTerminal: IndexedSeq[NonTerminal[S]] = finiteMarkovDecisionProcess.nonTerminalStates
    val (optimalValues, optimalActions) = nonTerminal.map { s: NonTerminal[S] =>
      val actions: Iterable[A] = finiteMarkovDecisionProcess.actions(s)
      val optimalAction: A = actions.maxBy(qValueFunction(s, _))
      val optimalValue: Double = qValueFunction(s, optimalAction)
      (s -> optimalValue, s.state -> optimalAction)
    }.unzip
    val valueFunction = optimalValues.toMap
    val policy = FiniteDeterministicPolicy(optimalActions.toMap)
    (valueFunction, policy)
  }
  
  def glieSarsaFiniteLearningRate[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)(_)
    
    TemporalDifference.glieSarsa(
      markovDecisionProcess = finiteMarkovDecisionProcess,
      initialStateDistribution = Choose(finiteMarkovDecisionProcess.nonTerminalStates),
      initialApproximation = Tabular[(NonTerminal[S], A)](countToWeight = learningRateFunction),
      gamma = gamma,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = maxEpisodeLength
    )
  }
  
  def glieSarsaLearningRateCorrectness[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int,
    numUpdates: Int
  ): Unit = {
    val nonTerminalStates = finiteMarkovDecisionProcess.nonTerminalStates
    
    val featureFunctions: Seq[((NonTerminal[S], A)) => Double] = nonTerminalStates.flatMap { s =>
      finiteMarkovDecisionProcess.actions(s).map { action =>
        x: (NonTerminal[S], A) => if ((x._1.state == s.state) && (x._2 == action)) 1.0 else 0.0
      }
    }
    
    val adamGradient = AdamGradient(
      learningRate = 0.05,
      decay1 = 0.9,
      decay2 = 0.999
    )
    
    val initialApproximation: QValueFunctionApproximation[S, A] =
      LinearFunctionApproximation.create[(NonTerminal[S], A)](
        featureFunctions = featureFunctions,
        adamGradient = adamGradient
      )
    
    val qValueFunctions = glieSarsaLearningRate(
      markovDecisionProcess = finiteMarkovDecisionProcess,
      initialStateDistribution = Choose(nonTerminalStates),
      gamma = gamma,
      initialApproximation = initialApproximation,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = maxEpisodeLength
    )
    
    val finalQValueFunction: QValueFunctionApproximation[S, A] = qValueFunctions.drop(numUpdates).next()
    val (valueFunction, policy) =
      getValueFunctionAndPolicyFromQValueFunction(finiteMarkovDecisionProcess, finalQValueFunction)
    
    val (trueValueFunction, truePolicy) = valueIterationResult(finiteMarkovDecisionProcess, gamma)
    
    val valueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${valueFunction(s)}%1.4f" }
      .mkString("\n")
    
    val trueValueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${trueValueFunction(s)}%1.4f" }
      .mkString("\n")
    
    val trueValues = DenseVector.apply(nonTerminalStates.toArray).map(trueValueFunction.apply)
    val pred = DenseVector.apply(nonTerminalStates.toArray).map(valueFunction.apply)
    val error = pred - trueValues
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"GLIE SARSA (Linear Approximation) Optimal Value Function with $numUpdates updates of length $maxEpisodeLength:\n$valueFunctionStr")
    logger.info(f"GLIE SARSA (Linear Approximation) Optimal Policy with $numUpdates updates of length $maxEpisodeLength:\n$policy")
    logger.info(f"True Optimal Value Function:\n$trueValueFunctionStr")
    logger.info(f"True Optimal Policy:\n$truePolicy")
    logger.info(f"RMSE: $rmse%1.4f")
  }
  
  def glieSarsaLearningRate[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    TemporalDifference.glieSarsa(
      markovDecisionProcess = markovDecisionProcess,
      initialStateDistribution = initialStateDistribution,
      initialApproximation = initialApproximation,
      gamma = gamma,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = maxEpisodeLength
    )
  }
  
  def glieSarsaLambdaFiniteLearningRate[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    lambda: Double,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)(_)
    
    TemporalDifferenceLambda.glieSarsaLambda(
      markovDecisionProcess = finiteMarkovDecisionProcess,
      initialStateDistribution = Choose(finiteMarkovDecisionProcess.nonTerminalStates),
      initialApproximation = Tabular[(NonTerminal[S], A)](countToWeight = learningRateFunction),
      gamma = gamma,
      lambda = lambda,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = maxEpisodeLength
    )
  }
  
  def glieSarsaLambdaLearningRate[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double,
    lambda: Double,
    epsilonFunction: Int => Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    TemporalDifferenceLambda.glieSarsaLambda(
      markovDecisionProcess = markovDecisionProcess,
      initialStateDistribution = initialStateDistribution,
      initialApproximation = initialApproximation,
      gamma = gamma,
      lambda = lambda,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = maxEpisodeLength
    )
  }
  
  def qLearningLearningRate[S, A](
    markovDecisionProcess: MarkovDecisionProcess[S, A],
    initialStateDistribution: NTStateDistribution[S],
    initialApproximation: QValueFunctionApproximation[S, A],
    gamma: Double,
    epsilon: Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
    val policyFromQValue: PolicyFromQValue[S, A] =
      (f: QValueFunctionApproximation[S, A], mdp: MarkovDecisionProcess[S, A]) =>
        MonteCarlo.epsilonGreedyPolicy(f, mdp, epsilon)
    
    TemporalDifference.qLearning(
      markovDecisionProcess = markovDecisionProcess,
      policyFromQValue = policyFromQValue,
      initialStateDistribution = initialStateDistribution,
      initialApproximation = initialApproximation,
      gamma = gamma,
      maxEpisodeLength = maxEpisodeLength
    )
  }
  
  def qLearningFiniteLearningRate[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    gamma: Double,
    epsilon: Double,
    maxEpisodeLength: Int
  ): Iterator[QValueFunctionApproximation[S, A]] = {
  
    val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)(_)
    
    val policyFromQValue: PolicyFromQValue[S, A] =
      (f: QValueFunctionApproximation[S, A], mdp: MarkovDecisionProcess[S, A]) =>
        MonteCarlo.epsilonGreedyPolicy(f, mdp, epsilon)
    
    TemporalDifference.qLearning(
      markovDecisionProcess = finiteMarkovDecisionProcess,
      policyFromQValue = policyFromQValue,
      initialStateDistribution = Choose(finiteMarkovDecisionProcess.nonTerminalStates),
      initialApproximation = Tabular[(NonTerminal[S], A)](countToWeight = learningRateFunction),
      gamma = gamma,
      maxEpisodeLength = maxEpisodeLength
    )
  }
  
  def qLearningFiniteLearningRateCorrectness[S, A](
    finiteMarkovDecisionProcess: FiniteMarkovDecisionProcess[S, A],
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    gamma: Double,
    epsilon: Double,
    maxEpisodeLength: Int,
    numUpdates: Int
  ): Unit = {
  
    val qValueFunctions = qLearningFiniteLearningRate(
      finiteMarkovDecisionProcess = finiteMarkovDecisionProcess,
      gamma = gamma,
      initialLearningRate = initialLearningRate,
      halfLife = halfLife,
      exponent = exponent,
      epsilon = epsilon,
      maxEpisodeLength = maxEpisodeLength
    )
  
    val finalQValueFunction: QValueFunctionApproximation[S, A] = qValueFunctions.drop(numUpdates).next()
    val (valueFunction, policy) =
      getValueFunctionAndPolicyFromQValueFunction(finiteMarkovDecisionProcess, finalQValueFunction)
  
    val (trueValueFunction, truePolicy) = valueIterationResult(finiteMarkovDecisionProcess, gamma)
  
    val nonTerminalStates = finiteMarkovDecisionProcess.nonTerminalStates
    val valueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${valueFunction(s)}%1.4f" }
      .mkString("\n")
  
    val trueValueFunctionStr = nonTerminalStates
      .map { s => f"Value for $s: ${trueValueFunction(s)}%1.4f" }
      .mkString("\n")
  
    val trueValues = DenseVector.apply(nonTerminalStates.toArray).map(trueValueFunction.apply)
    val pred = DenseVector.apply(nonTerminalStates.toArray).map(valueFunction.apply)
    val error = pred - trueValues
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"Q-Learning (Tabular) Optimal Value Function with $numUpdates updates of length $maxEpisodeLength:\n$valueFunctionStr")
    logger.info(f"Q-Learning (Tabular) Optimal Policy with $numUpdates updates of length $maxEpisodeLength:\n$policy")
    logger.info(f"True Optimal Value Function:\n$trueValueFunctionStr")
    logger.info(f"True Optimal Policy:\n$truePolicy")
    logger.info(f"RMSE: $rmse%1.4f")
    logger.info(f"Counts Map:\n${finalQValueFunction.asInstanceOf[Tabular[(NonTerminal[S], A)]].countsMapToString}")
  }
  
}
