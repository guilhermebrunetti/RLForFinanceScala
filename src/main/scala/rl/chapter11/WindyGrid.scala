package rl.chapter11

import java.time.LocalDateTime
import java.util.Locale

import breeze.numerics._
import WindyGrid._
import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming.{ValueFunction, valueIterationResult}
import rl.{FiniteDeterministicPolicy, FiniteMarkovDecisionProcess}
import rl.FiniteMarkovDecisionProcess.{StateActionMapping, processInputMap}
import rl.utils.{Categorical, ControlUtils}

/**
 * WindSpec specifies a random vertical wind for each column.
 * Each random vertical wind is specified by a (p1, p2) pair
 * where p1 specifies probability of Downward Wind (could take you
 * one step lower in row coordinate unless prevented by a block or
 * boundary) and p2 specifies probability of Upward Wind (could take
 * you onw step higher in column coordinate unless prevented by a
 * block or boundary). If one bumps against a block or boundary, one
 * incurs a bump cost and doesn't move. The remaining probability
 * 1- p1 - p2 corresponds to No Wind.
 */
class WindyGrid(
  val rows: Int, // number of grid rows
  val columns: Int, // number of grid columns
  val blocks: CellSet, // coordinates of block cells
  val terminals: CellSet, // coordinates of goal cells
  val wind: WindSpec, // spec of vertical random wind for the columns
  val bumpCost: Double // cost of bumping against block or boundary
) {
  require(validateSpec, "WindyGrid Spec is not valid")
  
  val blocksAndTerminals: Set[(Int, Int)] = blocks ++ terminals
  
  def validateSpec: Boolean = {
    val b1 = rows >= 2
    val b2 = columns >= 2
    val b3 = blocks.forall { case (r, c) => (0 <= r) && (r < rows) && (0 <= c) && (c < columns) }
    val b4 = terminals.nonEmpty
    val b5 = terminals.forall { case (r, c) =>
      (0 <= r) && (r < rows) && (0 <= c) && (c < columns) && !blocks.contains((r, c))
    }
    val b6 = wind.length == columns
    val b7 = wind.forall { case (p1, p2) =>
      (0.0 <= p1) && (p1 <= 1.0) && (0.0 <= p2) && (p2 <= 1.0) && (p1 + p2 <= 1.0)
    }
    val b8 = bumpCost > 0.0
    Seq(b1, b2, b3, b4, b5, b6, b7, b8).reduce(_ && _)
  }
  
  def printWindAndBumps(): Unit = {
    logger.info(s"Winds and Bump Cost")
    wind.zipWithIndex.foreach { case ((p1, p2), i) =>
      logger.info(f"Column $i: Down Prob = $p1, Up Prob = $p2")
    }
    logger.info(f"Bump Cost: $bumpCost")
  }
  
  def printValueFunctionAndPolicy(
    valueFunction: ValueFunction[Cell],
    policy: FiniteDeterministicPolicy[Cell, Move]
  ): Unit = {
    val zero: Double = 0.0
    val vfStringMap = valueFunction.map { case (k, v) => k.state -> f"${-v}%5.2f" } ++
      terminals.map { s => s -> f"$zero%5.2f" }.toMap ++
      blocks.map { s => s -> ("X" * 5) }
    logger.info(s"Value Function:")
    logger.info(f"   ${(0 until columns).map(j => f"$j%5d").mkString(" ")}")
    (0 until rows).reverse.foreach { i =>
      val cols = (0 until columns).map(j => vfStringMap((i, j))).mkString(" ")
      logger.info(f"$i%2d $cols")
    }
    
    val piStringMap = getAllNonTerminalStates.map { s => s -> possibleMoves(policy.actionForState(s)) }.toMap ++
      terminals.map { s => s -> "T" }.toMap ++
      blocks.map { s => s -> "X" }
    logger.info(s"Policy:")
    logger.info(f"   ${(0 until columns).map(j => f"$j%2d").mkString(" ")}")
    (0 until rows).reverse.foreach { i =>
      val cols = (0 until columns).map(j => piStringMap((i, j))).mkString("  ")
      logger.info(f"$i%2d  $cols")
    }
    
  }
  
  /**
   * returns all the non-terminal states
   */
  def getAllNonTerminalStates: CellSet = {
    (0 until rows)
      .flatMap(i => (0 until columns).map(j => (i, j)))
      .filter(!blocksAndTerminals.contains(_))
      .toSet
  }
  
  def getValueIterationValueFunctionAndPolicy: (ValueFunction[Cell], FiniteDeterministicPolicy[Cell, Move]) = {
    valueIterationResult(getFiniteMDP, gamma = 1.0)
  }
  
  def getGlieSarsaValueFunctionAndPolicy(
    epsilonFunction: Int => Double,
    learningRate: Double,
    exponent: Double,
    halfLife: Double,
    episodeLength: Int,
    numUpdates: Int
  ): (ValueFunction[Cell], FiniteDeterministicPolicy[Cell, Move]) = {
    
    val finiteMarkovDecisionProcess = getFiniteMDP
    
    val qValueFunctions = ControlUtils.glieSarsaFiniteLearningRate(
      finiteMarkovDecisionProcess = finiteMarkovDecisionProcess,
      gamma = 1.0,
      initialLearningRate = learningRate,
      exponent = exponent,
      halfLife = halfLife,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = episodeLength
    )
    
    val qValueFunction = qValueFunctions.drop(numUpdates).next()
    ControlUtils.getValueFunctionAndPolicyFromQValueFunction(finiteMarkovDecisionProcess, qValueFunction)
  }
  
  def getGlieSarsaLambdaValueFunctionAndPolicy(
    epsilonFunction: Int => Double,
    learningRate: Double,
    exponent: Double,
    halfLife: Double,
    lambda: Double,
    episodeLength: Int,
    numUpdates: Int
  ): (ValueFunction[Cell], FiniteDeterministicPolicy[Cell, Move]) = {
    
    val finiteMarkovDecisionProcess = getFiniteMDP
    
    val qValueFunctions = ControlUtils.glieSarsaLambdaFiniteLearningRate(
      finiteMarkovDecisionProcess = finiteMarkovDecisionProcess,
      gamma = 1.0,
      lambda = lambda,
      initialLearningRate = learningRate,
      exponent = exponent,
      halfLife = halfLife,
      epsilonFunction = epsilonFunction,
      maxEpisodeLength = episodeLength
    )
    
    val qValueFunction = qValueFunctions.drop(numUpdates).next()
    ControlUtils.getValueFunctionAndPolicyFromQValueFunction(finiteMarkovDecisionProcess, qValueFunction)
  }
  
  /**
   *
   * @return the FiniteMarkovDecision object for this windy grid problem
   */
  def getFiniteMDP: FiniteMarkovDecisionProcess[Cell, Move] = {
    val inputMap = getAllNonTerminalStates.map { s => s -> getTransitionProbabilities(s) }.toMap
    
    new FiniteMarkovDecisionProcess[Cell, Move] {
      override def stateActionMap: StateActionMapping[(Int, Int), (Int, Int)] = processInputMap(inputMap)
    }
  }
  
  /**
   * Given a non-terminal state, return a map whose
   * keys are the valid actions (moves) from the given state
   * and the corresponding values are the associated probabilities
   * (following that move) of the (nextState, reward) pairs.
   * The probabilities are determined from the wind probabilities
   * of the column one is in after the move. Note that if one moves
   * to a goal cell (terminal state), then one ends up in that
   * goal cell with 100% probability (i.e., no wind exposure in a
   * goal cell).
   */
  def getTransitionProbabilities(ntState: Cell): Map[Move, Categorical[(Cell, Double)]] = {
    getActionsAndNextStates(ntState)
      .toMap
      .view
      .mapValues { cell =>
        if (terminals.contains(cell)) {
          Categorical(Map((cell, -1.0) -> 1.0))
        }
        else {
          val (r, c) = cell
          val downCell = (r - 1, c)
          val upCell = (r + 1, c)
          val (downProb, upProb) = wind(c)
          val stayProb = 1.0 - downProb - upProb
          val bumpDownProb = downProb * (1 - isValidStateInt(downCell))
          val bumpUpProb = upProb * (1 - isValidStateInt(upCell))
          val inputMap = Map(
            (cell, -1.0) -> stayProb,
            (downCell, -1.0) -> downProb,
            (upCell, -1.0) -> upProb,
            (cell, -1.0 - bumpCost) -> (bumpDownProb + bumpUpProb)
          ).filter { case ((c, _), p) => isValidState(c) && p > 0.0 }
          Categorical(inputMap)
        }
      }.toMap
  }
  
  /**
   * Given a non-terminal state, returns the set of all possible
   * (action, next_state) pairs
   */
  def getActionsAndNextStates(ntState: Cell): Set[(Move, Cell)] = {
    possibleMoves.keySet.map { action => (action, addMoveToCell(ntState, action))
    }.filter { case (_, s) => isValidState(s) }
  }
  
  private def isValidStateInt(cell: Cell): Int = if (isValidState(cell)) 1 else 0
  
  /**
   * Checks if a cell is a valid state of the MDP
   */
  def isValidState(cell: Cell): Boolean = {
    val (r, c) = cell
    (0 <= r) && (r < rows) && (0 <= c) && (c < columns) && !blocks.contains(cell)
  }
  
  def getQLearningValueFunctionAndPolicy(
    epsilon: Double,
    learningRate: Double,
    exponent: Double,
    halfLife: Double,
    episodeLength: Int,
    numUpdates: Int
  ): (ValueFunction[Cell], FiniteDeterministicPolicy[Cell, Move]) = {
    
    val finiteMarkovDecisionProcess = getFiniteMDP
    
    val qValueFunctions = ControlUtils.qLearningFiniteLearningRate(
      finiteMarkovDecisionProcess = finiteMarkovDecisionProcess,
      gamma = 1.0,
      initialLearningRate = learningRate,
      exponent = exponent,
      halfLife = halfLife,
      epsilon = epsilon,
      maxEpisodeLength = episodeLength
    )
    
    val qValueFunction = qValueFunctions.drop(numUpdates).next()
    ControlUtils.getValueFunctionAndPolicyFromQValueFunction(finiteMarkovDecisionProcess, qValueFunction)
  }
  
}

object WindyGrid {
  
  type Cell = (Int, Int)
  type CellSet = Set[Cell]
  type Move = (Int, Int)
  type WindSpec = Seq[(Double, Double)]
  val logger: Logger = Logger("WindyGrid")
  val possibleMoves: Map[Move, String] = Map(
    (-1, 0) -> "D",
    (1, 0) -> "U",
    (0, -1) -> "L",
    (0, 1) -> "R"
  )
  
  def addMoveToCell(cell: Cell, move: Move): Cell = {
    val (r, c) = cell
    val (m1, m2) = move
    (r + m1, c + m2)
  }
  
  def apply(
    rows: Int,
    columns: Int,
    blocks: CellSet,
    terminals: CellSet,
    wind: WindSpec,
    bumpCost: Double
  ): WindyGrid = new WindyGrid(rows, columns, blocks, terminals, wind, bumpCost)
  
}

object WindyGridApp extends App {
  
  val logger: Logger = Logger("WindyGridApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val windyGrid = WindyGrid(
    rows = 5,
    columns = 5,
    blocks = Set((0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (4, 0)),
    terminals = Set((3, 4)),
    wind = Seq((0.0, 0.9), (0.0, 0.8), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)),
    bumpCost = 4.0
  )
  
  windyGrid.printWindAndBumps()
  val (trueValueFunction, truePolicy) = windyGrid.getValueIterationValueFunctionAndPolicy
  logger.info("Value Iteration:")
  windyGrid.printValueFunctionAndPolicy(trueValueFunction, truePolicy)
  
  val epsilonFunction: Int => Double = (k: Int) => pow(k, -1.0)
  val learningRate = 0.03
  val exponent = 1.0
  val halfLife = 1.0e4
  val numUpdates = 1.0e5.toInt
  val episodeLength = 1.0e8.toInt
  val lambda = 0.30
  val epsilon = 0.2
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val (sarsaValueFunction, sarsaPolicy) = windyGrid.getGlieSarsaValueFunctionAndPolicy(
    epsilonFunction = epsilonFunction,
    learningRate = learningRate,
    exponent = exponent,
    halfLife = halfLife,
    episodeLength = episodeLength,
    numUpdates = numUpdates
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info("GLIE-SARSA:")
  windyGrid.printValueFunctionAndPolicy(sarsaValueFunction, sarsaPolicy)
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val (sarsaLambdaValueFunction, sarsaLambdaPolicy) = windyGrid.getGlieSarsaLambdaValueFunctionAndPolicy(
    epsilonFunction = epsilonFunction,
    learningRate = learningRate,
    lambda = lambda,
    exponent = exponent,
    halfLife = halfLife,
    episodeLength = episodeLength,
    numUpdates = numUpdates
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(f"GLIE-SARSA Lambda (lambda=$lambda):")
  windyGrid.printValueFunctionAndPolicy(sarsaLambdaValueFunction, sarsaLambdaPolicy)
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val (qLearningValueFunction, qLearningPolicy) = windyGrid.getQLearningValueFunctionAndPolicy(
    epsilon = epsilon,
    learningRate = learningRate,
    exponent = exponent,
    halfLife = halfLife,
    episodeLength = episodeLength,
    numUpdates = numUpdates
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")

  logger.info("Q Learning:")
  windyGrid.printValueFunctionAndPolicy(qLearningValueFunction, qLearningPolicy)
}
