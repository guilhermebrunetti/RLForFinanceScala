package rl.chapter10

import java.util.Locale

import com.typesafe.scalalogging.Logger
import rl.{FiniteMarkovRewardProcess, NonTerminal}
import rl.FiniteMarkovRewardProcess.{RewardTransition, processInputMap}
import rl.utils.{Categorical, FiniteDistribution, PredictionUtils}

class RandomWalkMRP(
  val barrier: Int,
  val probability: Double,
) extends FiniteMarkovRewardProcess[Int] {
  
  override def sortingFunction(x: NonTerminal[Int], y: NonTerminal[Int]): Boolean = x.state <= y.state
  
  override def transitionRewardMap: RewardTransition[Int] = processInputMap(getTransitionMap)
  
  def getTransitionMap: Map[Int, FiniteDistribution[(Int, Double)]] = {
    (1 until barrier).map { i =>
      i -> Categorical(Map(
        (i + 1, if (i < barrier - 1) 0.0 else 1.0) -> probability,
        (i - 1, 0.0) -> (1.0 - probability)
      ))
    }.toMap
  }
}

object RandomWalkMRP {
  def apply(
    barrier: Int,
    probability: Double
  ): RandomWalkMRP = new RandomWalkMRP(barrier, probability)
}

object RandomWalkMRPApp extends App {
  
  val logger: Logger = Logger("RandomWalkMRPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val barrier: Int = 10
  val probability: Double = 0.4
  val randomWalkMRP = RandomWalkMRP(barrier, probability)
  
  val initialLearningRate = 0.01
  val halfLife = 1.0e5
  val exponent = 1.0
  val gamma = 1.0
  val lambda = 0.3
  
  val nonTerminalStates = randomWalkMRP.nonTerminalStates
  val trueValueFunction = randomWalkMRP.valueFunctionVector(gamma)
  val trueValueFunctionStr = randomWalkMRP.valueFunctionToString(gamma)
  
  val episodeLengthTolerance = 1.0e-6
  val numEpisodes = 700
  val episodeLength = 20
  
  PredictionUtils.mcFiniteEqualWeightsCorrectness(
    finiteMarkovRewardProcess = randomWalkMRP,
    gamma = gamma,
    episodeLengthTolerance = episodeLengthTolerance,
    numEpisodes = numEpisodes
  )
  
  PredictionUtils.mcFinitePredictionLearningRateCorrectness(
    finiteMarkovRewardProcess = randomWalkMRP,
    gamma = gamma,
    episodeLengthTolerance = episodeLengthTolerance,
    numEpisodes = numEpisodes,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent
  )
  
  PredictionUtils.tdFiniteLearningRateCorrectness(
    finiteMarkovRewardProcess = randomWalkMRP,
    gamma = gamma,
    episodeLength = episodeLength,
    numEpisodes = numEpisodes * episodeLength,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent
  )
  
  PredictionUtils.tdLambdaFiniteLearningRateCorrectness(
    finiteMarkovRewardProcess = randomWalkMRP,
    gamma = gamma,
    lambda = lambda,
    episodeLength = episodeLength,
    numEpisodes = numEpisodes * episodeLength,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent
  )
  
}
