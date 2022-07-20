package rl.utils

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming.{NTStateDistribution, ValueFunctionApproximation}
import rl.DynamicProgramming.ValueFunction
import rl.Tabular.learningRateSchedule
import rl._


object PredictionUtils {
  
  val logger: Logger = Logger("PredictionUtils")
  
  def mcPredictionLearningRate[S](
    markovRewardProcess: MarkovRewardProcess[S],
    gamma: Double,
    initialStateDistribution: NTStateDistribution[S],
    episodeLengthTolerance: Double,
    initialApproximation: ValueFunctionApproximation[S]
  ): Iterator[ValueFunctionApproximation[S]] = {
    val episodes = mrpEpisodeStream(markovRewardProcess, initialStateDistribution)
    MonteCarlo.mcPrediction(
      traces = episodes,
      initialApproximation = initialApproximation,
      gamma = gamma,
      episodeLengthTolerance = episodeLengthTolerance
    )
  }
  
  def mrpEpisodeStream[S](
    markovRewardProcess: MarkovRewardProcess[S],
    initialStateDistribution: NTStateDistribution[S]
  ): Iterable[Iterable[TransitionStep[S]]] = {
    markovRewardProcess.rewardTraces(initialStateDistribution)
  }
  
  def tdPredictionLearningRate[S](
    markovRewardProcess: MarkovRewardProcess[S],
    gamma: Double,
    initialStateDistribution: NTStateDistribution[S],
    episodeLength: Int,
    initialApproximation: ValueFunctionApproximation[S]
  ): Iterator[ValueFunctionApproximation[S]] = {
    val episodes = mrpEpisodeStream(markovRewardProcess, initialStateDistribution)
    val tdExperiences = unitExperiencesFromEpisodes(episodes, episodeLength)
    TemporalDifference.tdPrediction(
      transitions = tdExperiences,
      initialApproximation = initialApproximation,
      gamma = gamma
    )
  }
  
  def unitExperiencesFromEpisodes[S](
    episodes: Iterable[Iterable[TransitionStep[S]]],
    episodeLength: Int
  ): Iterable[TransitionStep[S]] = {
    episodes.flatMap(_.take(episodeLength))
  }
  
  def tdLambdaPredictionLearningRate[S](
    markovRewardProcess: MarkovRewardProcess[S],
    gamma: Double,
    lambda: Double,
    initialStateDistribution: NTStateDistribution[S],
    episodeLength: Int,
    initialApproximation: ValueFunctionApproximation[S]
  ): Iterator[ValueFunctionApproximation[S]] = {
    val episodes = mrpEpisodeStream(markovRewardProcess, initialStateDistribution)
    val curtailedEpisodes = episodes.map(_.take(episodeLength).toIndexedSeq)
    TemporalDifferenceLambda.tdLambdaPrediction(
      traces = curtailedEpisodes,
      initialApproximation = initialApproximation,
      gamma = gamma,
      lambda = lambda
    )
  }
  
  def tdLambdaFinitePredictionLearningRate[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    lambda: Double,
    episodeLength: Int,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Iterator[ValueFunctionApproximation[S]] = {
    
    val episodes = finiteMrpEpisodeStream(finiteMarkovRewardProcess)
    val curtailedEpisodes = episodes.map(_.take(episodeLength))
    val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)(_)
    
    TemporalDifferenceLambda.tdLambdaPrediction(
      traces = curtailedEpisodes,
      initialApproximation = Tabular(
        valuesMap = initialValueFunction,
        countToWeight = learningRateFunction
      ),
      gamma = gamma,
      lambda = lambda
    )
    
  }
  
  def mcFiniteEqualWeightsCorrectness[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    episodeLengthTolerance: Double,
    numEpisodes: Int,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Unit = {
    val mcValueFunctions = mcFinitePredictionEqualWeights(
        finiteMarkovRewardProcess,
        gamma,
        episodeLengthTolerance,
        initialValueFunction
      ).take(numEpisodes).toSeq
    
    val finalValueFunction: ValueFunctionApproximation[S] = mcValueFunctions.last
    val finalVFStr = finiteMarkovRewardProcess.nonTerminalStates
      .map { s => f"Value for $s: ${finalValueFunction(s)}%1.4f" }
      .mkString("\n")
    val trueValueFunctionStr = finiteMarkovRewardProcess.valueFunctionToString(gamma)
    val trueValueFunction = finiteMarkovRewardProcess.valueFunctionVector(gamma)
    val pred: DenseVector[Double] = finalValueFunction.evaluate(finiteMarkovRewardProcess.nonTerminalStates)
    val error = pred - trueValueFunction
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"Equal-Weights-MC Value Function with $numEpisodes episodes:\n$finalVFStr")
    logger.info(f"True Value Function:\n$trueValueFunctionStr")
    logger.info(f"RMSE: $rmse%1.4f")
    logger.info(f"Counts Map:\n${finalValueFunction.asInstanceOf[Tabular[NonTerminal[S]]].countsMapToString}")
  }
  
  def mcFinitePredictionEqualWeights[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    episodeLengthTolerance: Double,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Iterator[ValueFunctionApproximation[S]] = {
    val episodes = finiteMrpEpisodeStream(finiteMarkovRewardProcess)
    MonteCarlo.mcPrediction(
      traces = episodes,
      initialApproximation = Tabular(valuesMap = initialValueFunction),
      gamma = gamma,
      episodeLengthTolerance = episodeLengthTolerance
    )
  }
  
  def mcFinitePredictionLearningRateCorrectness[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    episodeLengthTolerance: Double,
    numEpisodes: Int,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Unit = {
    val mcValueFunctions = mcFinitePredictionLearningRate(
        finiteMarkovRewardProcess,
        gamma,
        episodeLengthTolerance,
        initialLearningRate,
        halfLife,
        exponent,
        initialValueFunction
      )
    
    val finalValueFunction: ValueFunctionApproximation[S] = mcValueFunctions.drop(numEpisodes).next()
    val finalVFStr = finiteMarkovRewardProcess.nonTerminalStates
      .map { s => f"Value for $s: ${finalValueFunction(s)}%1.4f" }
      .mkString("\n")
    val trueValueFunctionStr = finiteMarkovRewardProcess.valueFunctionToString(gamma)
    val trueValueFunction = finiteMarkovRewardProcess.valueFunctionVector(gamma)
    val pred: DenseVector[Double] = finalValueFunction.evaluate(finiteMarkovRewardProcess.nonTerminalStates)
    val error = pred - trueValueFunction
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"Decaying-Learning-Rate-MC Value Function with $numEpisodes episodes:\n$finalVFStr")
    logger.info(f"True Value Function:\n$trueValueFunctionStr")
    logger.info(f"RMSE: $rmse%1.4f")
    logger.info(f"Counts Map:\n${finalValueFunction.asInstanceOf[Tabular[NonTerminal[S]]].countsMapToString}")
  }
  
  def mcFinitePredictionLearningRate[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    episodeLengthTolerance: Double,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Iterator[ValueFunctionApproximation[S]] = {
    val episodes = finiteMrpEpisodeStream(finiteMarkovRewardProcess)
    val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)
    
    MonteCarlo.mcPrediction(
      traces = episodes,
      initialApproximation = Tabular(
        valuesMap = initialValueFunction,
        countToWeight = learningRateFunction
      ),
      gamma = gamma,
      episodeLengthTolerance = episodeLengthTolerance
    )
  }
  
  def finiteMrpEpisodeStream[S](
    finiteMRP: FiniteMarkovRewardProcess[S]
  ): Iterable[Iterable[TransitionStep[S]]] = {
    mrpEpisodeStream(finiteMRP, Choose(finiteMRP.nonTerminalStates))
  }
  
  def tdFiniteLearningRateCorrectness[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    episodeLength: Int,
    numEpisodes: Int,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Unit = {
    val tdValueFunctions = tdFinitePredictionLearningRate(
        finiteMarkovRewardProcess,
        gamma,
        episodeLength,
        initialLearningRate,
        halfLife,
        exponent,
        initialValueFunction
      )
    
    val finalValueFunction: ValueFunctionApproximation[S] = tdValueFunctions.drop(numEpisodes).next()
    val finalVFStr = finiteMarkovRewardProcess.nonTerminalStates
      .map { s => f"Value for $s: ${finalValueFunction(s)}%1.4f" }
      .mkString("\n")
    val trueValueFunctionStr = finiteMarkovRewardProcess.valueFunctionToString(gamma)
    val trueValueFunction = finiteMarkovRewardProcess.valueFunctionVector(gamma)
    val pred: DenseVector[Double] = finalValueFunction.evaluate(finiteMarkovRewardProcess.nonTerminalStates)
    val error = pred - trueValueFunction
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"Decaying-Learning-Rate-TD Value Function with $numEpisodes episodes of length $episodeLength:\n$finalVFStr")
    logger.info(f"True Value Function:\n$trueValueFunctionStr")
    logger.info(f"RMSE: $rmse%1.4f")
    logger.info(f"Counts Map:\n${finalValueFunction.asInstanceOf[Tabular[NonTerminal[S]]].countsMapToString}")
  }
  
  def tdFinitePredictionLearningRate[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    episodeLength: Int,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Iterator[ValueFunctionApproximation[S]] = {
    val episodes = finiteMrpEpisodeStream(finiteMarkovRewardProcess)
    val tdExperiences = unitExperiencesFromEpisodes(episodes, episodeLength)
    val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)(_)
    
    TemporalDifference.tdPrediction(
      transitions = tdExperiences,
      initialApproximation = Tabular(
        valuesMap = initialValueFunction,
        countToWeight = learningRateFunction
      ),
      gamma = gamma
    )
  }
  
  def tdLambdaFiniteLearningRateCorrectness[S](
    finiteMarkovRewardProcess: FiniteMarkovRewardProcess[S],
    gamma: Double,
    lambda: Double,
    episodeLength: Int,
    numEpisodes: Int,
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double,
    initialValueFunction: ValueFunction[S] = Map.empty[NonTerminal[S], Double]
  ): Unit = {
    val tdValueFunctions = tdLambdaFinitePredictionLearningRate(
        finiteMarkovRewardProcess,
        gamma,
        lambda,
        episodeLength,
        initialLearningRate,
        halfLife,
        exponent,
        initialValueFunction
      )
    
    val finalValueFunction: ValueFunctionApproximation[S] = tdValueFunctions.drop(numEpisodes).next()
    val finalVFStr = finiteMarkovRewardProcess.nonTerminalStates
      .map { s => f"Value for $s: ${finalValueFunction(s)}%1.4f" }
      .mkString("\n")
    val trueValueFunctionStr = finiteMarkovRewardProcess.valueFunctionToString(gamma)
    val trueValueFunction = finiteMarkovRewardProcess.valueFunctionVector(gamma)
    val pred: DenseVector[Double] = finalValueFunction.evaluate(finiteMarkovRewardProcess.nonTerminalStates)
    val error = pred - trueValueFunction
    val rmse = sqrt(mean(pow(error, 2)))
    logger.info(f"Decaying-Learning-Rate-TD-Lambda Value Function with $numEpisodes episodes of length $episodeLength:\n$finalVFStr")
    logger.info(f"True Value Function:\n$trueValueFunctionStr")
    logger.info(f"RMSE: $rmse%1.4f")
    logger.info(f"Counts Map:\n${finalValueFunction.asInstanceOf[Tabular[NonTerminal[S]]].countsMapToString}")
  }
  
}
