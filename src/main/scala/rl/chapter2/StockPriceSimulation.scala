package rl.chapter2

import java.util.Locale

import breeze.linalg._
import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.utils.Utils

trait PriceState {
  def price: Int
}

trait PriceProcess[T <: PriceState] {
  def nextState(state: T): T
  
  def priceTraces(initialState: T, timeSteps: Int, numTraces: Int): DenseMatrix[Int] = {
    val traces: Seq[Seq[T]] = stateTraces(initialState, timeSteps, numTraces)
    val vectors: Seq[DenseVector[Int]] = traces.map(sq => DenseVector(sq.map(_.price): _*))
    DenseMatrix(vectors: _*)
  }
  
  def stateTraces(initialState: T, timeSteps: Int, numTraces: Int): Seq[Seq[T]] = {
    (1 to numTraces).map(_ => simulateStream(initialState).take(timeSteps + 1))
  }
  
  def simulateStream(initialState: T): LazyList[T] = {
    LazyList.unfold(initialState) { state => Some((state, nextState(state))) }
  }
}

case class PriceState1(price: Int) extends PriceState

class Process1(val levelParam: Int, val alpha1: Double = 0.25)
  extends PriceProcess[PriceState1] {
  require(alpha1 >= 0, s"Alpha1 needs to be positive, got value $alpha1")
  
  override def nextState(state: PriceState1): PriceState1 = {
    val upProb: Double = upProbability(state)
    val upMove: Int = distributions.Binomial(1, upProb).draw()
    PriceState1(state.price + upMove * 2 - 1)
  }
  
  def upProbability(state: PriceState1): Double = {
    Utils.getLogisticFunction(alpha1)(levelParam - state.price)
  }
  
}

case class PriceState2(price: Int, previousMoveUp: Option[Boolean] = None)
  extends PriceState

class Process2(val alpha2: Double = 0.75)
  extends PriceProcess[PriceState2] {
  require(alpha2 >= 0, s"Alpha2 needs to be positive, got value $alpha2")
  require(alpha2 <= 1, s"Alpha2 needs smaller than 1, got value $alpha2")
  
  override def nextState(state: PriceState2): PriceState2 = {
    val upProb: Double = upProbability(state)
    val upMove: Int = distributions.Binomial(1, upProb).draw()
    PriceState2(state.price + upMove * 2 - 1, Some(upMove == 1))
  }
  
  def upProbability(state: PriceState2): Double = {
    val aux = state.previousMoveUp.map {
      case true => -1.0
      case false => 1.0
    }.getOrElse(0.0)
    0.5 * (1.0 + alpha2 * aux)
  }
  
}

case class PriceState3(initialPrice: Int, numUpMoves: Int = 0, numDownMoves: Int = 0)
  extends PriceState {
  def price: Int = initialPrice + numUpMoves - numDownMoves
}

class Process3(val alpha3: Double = 1.0)
  extends PriceProcess[PriceState3] {
  require(alpha3 >= 0, s"Alpha2 needs to be positive, got value $alpha3")
  
  override def nextState(state: PriceState3): PriceState3 = {
    val upProb: Double = upProbability(state)
    val upMove: Int = distributions.Binomial(1, upProb).draw()
    state.copy(
      numUpMoves = state.numUpMoves + upMove,
      numDownMoves = state.numDownMoves + 1 - upMove
    )
  }
  
  def upProbability(state: PriceState3): Double = {
    val downMoves: Double = state.numDownMoves.toDouble
    val total: Double = state.numUpMoves.toDouble + downMoves
    val ratio: Double = downMoves / total
    if (total > 0) Utils.getUnitSigmoidFunction(alpha3)(ratio) else 0.5
  }
}

object StockPriceSimulation extends App {
  val logger: Logger = Logger("StockPriceSimulation")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val initialPrice = 100
  val levelParam = 100
  val alpha1 = 0.25
  val alpha2 = 0.75
  val alpha3 = 1.0
  val timeSteps = 10
  val numTraces = 7
  
  val initialState1 = PriceState1(initialPrice)
  val process1 = new Process1(levelParam, alpha1)
  
  val initialState2 = PriceState2(initialPrice)
  val process2 = new Process2(alpha2)
  
  val initialState3 = PriceState3(initialPrice = initialPrice)
  val process3 = new Process3(alpha3)
  
  val singleTrajectory1 = process1
    .simulateStream(initialState1)
    .map(_.price)
    .take(timeSteps)
    .toList
  
  val singleTrajectory2 = process2
    .simulateStream(initialState2)
    .map(_.price)
    .take(timeSteps)
    .toList
  
  val singleTrajectory3 = process3
    .simulateStream(initialState3)
    .map(_.price)
    .take(timeSteps)
    .toList
  
  logger.info(s"Process1 Price trajectory:\n$singleTrajectory1")
  logger.info(s"Process2 Price trajectory:\n$singleTrajectory2")
  logger.info(s"Process3 Price trajectory:\n$singleTrajectory3")
  
  val priceTraces1 = process1.priceTraces(initialState1, timeSteps, numTraces)
  val priceTraces2 = process2.priceTraces(initialState2, timeSteps, numTraces)
  val priceTraces3 = process3.priceTraces(initialState3, timeSteps, numTraces)
  
  logger.info(s"Sample traces from Process1:\n$priceTraces1")
  logger.info(s"Sample traces from Process2:\n$priceTraces2")
  logger.info(s"Sample traces from Process3:\n$priceTraces3")
  
}


