package rl

import rl.utils.Categorical

import scala.annotation.tailrec

class ExperienceReplayMemory[T](
  val savedTransitions: Seq[T] = Seq.empty,
  val timeWeightsFunction: Int => Double = _ => 1.0,
  val weights: Seq[Double] = Seq.empty,
  val weightsSum: Double = 0.0
) {
  require(
    savedTransitions.size == weights.size,
    s"savedTransitions (${savedTransitions.size}) and weights (${weights.size}) should have same size."
  )
  
  def numTransitions: Int = savedTransitions.size
  
  def addData(transition: T): ExperienceReplayMemory[T] = {
    val newWeight = timeWeightsFunction(numTransitions)
    ExperienceReplayMemory(
      this.savedTransitions :+ transition,
      this.timeWeightsFunction,
      this.weights :+ newWeight,
      this.weightsSum + newWeight
    )
  }
  
  def sampleMiniBatch(miniBatchSize: Int): Seq[T] = {
    val normalizedWeights = weights.reverse.map(_ / weightsSum)
    val distribution = Categorical.fromIterable(savedTransitions.zip(normalizedWeights))
    distribution.samples(miniBatchSize)
  }
  
  def replay(
    transitions: Iterable[T],
    miniBatchSize: Int
  ): Iterable[Seq[T]] = {
    ExperienceReplayMemory.replay(this, transitions, miniBatchSize)
  }
  
}

object ExperienceReplayMemory {
  
  def apply[T](
    savedTransitions: Seq[T] = Seq.empty,
    timeWeightsFunction: Int => Double = _ => 1.0,
    weights: Seq[Double] = Seq.empty,
    weightsSum: Double = 0.0
  ): ExperienceReplayMemory[T] =
    new ExperienceReplayMemory(savedTransitions, timeWeightsFunction, weights, weightsSum)
    
  @tailrec def replay[T](
    experienceReplayMemory: ExperienceReplayMemory[T],
    transitions: Iterable[T],
    miniBatchSize: Int,
    miniBatches: Seq[Seq[T]] = Seq.empty
  ): Iterable[Seq[T]] = {
    transitions.size match {
      case 0 => miniBatches ++ LazyList.continually(experienceReplayMemory.sampleMiniBatch(miniBatchSize))
      case _ =>
        val nextExperienceReplay = experienceReplayMemory.addData(transitions.head)
        val miniBatch = nextExperienceReplay.sampleMiniBatch(miniBatchSize)
        replay(nextExperienceReplay, transitions.tail, miniBatchSize, miniBatches :+ miniBatch)
    }
  }
}
