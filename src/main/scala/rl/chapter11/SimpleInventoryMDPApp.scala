package rl.chapter11

import java.time.LocalDateTime
import java.util.Locale

import breeze.numerics._
import com.typesafe.scalalogging.Logger
import rl.chapter3.SimpleInventoryMDPCap
import rl.utils.ControlUtils

object SimpleInventoryMDPApp extends App {
  
  val logger: Logger = Logger("SimpleInventoryMDPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val capacity = 2
  val poissonLambda = 1.0
  val holdingCost = 1.0
  val stockoutCost = 10.0
  
  val gamma = 0.9
  
  val SIMDPCap = SimpleInventoryMDPCap(capacity, poissonLambda, holdingCost, stockoutCost)
  
  val episodeLengthTolerance = 1.0e-5
  val mcNumEpisodes = 10000
  val tdNumEpisodes = 10000
  val episodeLength = 100
  val initialLearningRate = 0.1
  val halfLife = 1.0e4
  val exponent = 1.0
  val qLearningEpsilon = 0.2
  
  val lambda = 0.3
  
  def epsilonFunction(k: Int): Double = pow(k, -0.5)
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  ControlUtils.glieMCFiniteLearningRateCorrectness(
    finiteMarkovDecisionProcess = SIMDPCap,
    gamma = gamma,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent,
    epsilonFunction = epsilonFunction,
    episodeLengthTolerance = episodeLengthTolerance,
    numEpisodes = mcNumEpisodes
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  ControlUtils.glieSarsaFiniteLearningRateCorrectness(
    finiteMarkovDecisionProcess = SIMDPCap,
    gamma = gamma,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent,
    epsilonFunction = epsilonFunction,
    maxEpisodeLength = episodeLength,
    numUpdates = tdNumEpisodes * episodeLength
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  ControlUtils.qLearningFiniteLearningRateCorrectness(
    finiteMarkovDecisionProcess = SIMDPCap,
    gamma = gamma,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent,
    epsilon = qLearningEpsilon,
    maxEpisodeLength = episodeLength,
    numUpdates = tdNumEpisodes * episodeLength
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
}
