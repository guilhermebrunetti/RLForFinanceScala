package rl.chapter10

import java.util.Locale

import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming.ValueFunctionApproximation
import rl.FiniteMarkovRewardProcess.RewardTransition
import rl.utils.{Categorical, Choose}
import rl._
import ExperienceReplayUtils._

object ExperienceReplayUtils {
  
  def getFixedEpisodesFromStateRewards[S](
    stateRewardTraces: Iterable[Seq[(S, Double)]],
    terminalState: S
  ): Iterable[Seq[TransitionStep[S]]] = {
    stateRewardTraces.map { trace =>
      val initSeq: Seq[TransitionStep[S]] = if (trace.tail.nonEmpty) {
        trace.init.zip(trace.tail).map { case ((state, reward), (nextState, _)) =>
          TransitionStepMRP(NonTerminal(state), NonTerminal(nextState), reward)
        }
      } else Seq.empty[TransitionStep[S]]
      val (lastState, lastReward) = trace.last
      val lastValue = TransitionStepMRP(NonTerminal(lastState), Terminal(terminalState), lastReward)
      initSeq :+ lastValue
    }
  }
  
  def getReturnStepsFromFixedEpisodes[S](
    fixedEpisodes: Iterable[Seq[TransitionStep[S]]],
    gamma: Double,
    tolerance: Double = 1.0e-8
  ): Iterable[ReturnStep[S]] = {
    fixedEpisodes.flatMap { episodes => Returns.returns(episodes, gamma, tolerance) }
  }
  
  def getMeanReturnsFromReturnSteps[S](
    returnSteps: Iterable[ReturnStep[S]]
  ): Map[NonTerminal[S], Double] = {
    
    returnSteps.groupMap(_.state)(_.returns).view.mapValues(mean(_)).toMap
  }
  
  def getEpisodeStream[S](
    fixedEpisodes: Iterable[Seq[TransitionStep[S]]]
  ): LazyList[Seq[TransitionStep[S]]] = {
    val dist: Choose[Seq[TransitionStep[S]]] = Choose(fixedEpisodes)
    LazyList.continually(dist.sample)
  }
  
  def mcPrediction[S](
    episodesStream: Iterable[Seq[TransitionStep[S]]],
    gamma: Double,
    numEpisodes: Int,
    episodeLengthTolerance: Double = 1.0e-6
  ): Map[NonTerminal[S], Double] = {
    val valueApproximation: ValueFunctionApproximation[S] = MonteCarlo
      .mcPrediction(episodesStream, Tabular[NonTerminal[S]](), gamma, episodeLengthTolerance)
      .drop(numEpisodes)
      .next()
    
    valueApproximation.asInstanceOf[Tabular[NonTerminal[S]]].valuesMap
  }
  
  def fixedExperiencesFromFixedEpisodes[S](
    fixedEpisodes: Iterable[Seq[TransitionStep[S]]]
  ): Iterable[TransitionStep[S]] = {
    fixedEpisodes.flatten
  }
  
  def finiteMRP[S](
    fixedExperiences: Iterable[TransitionStep[S]]
  ): FiniteMarkovRewardProcess[S] = {
    
    val groupedMap: Map[NonTerminal[S], Iterable[(State[S], Double)]] =
      fixedExperiences.groupMap(_.state)(_.nextStateAndReward)
    
    val rewardMap: Map[NonTerminal[S], Categorical[(State[S], Double)]] = groupedMap.view.mapValues { stateRewards =>
      val n = stateRewards.size.toDouble
      val normalizedValues = stateRewards.map { (_, 1.0) }
      Categorical.fromIterable(stateRewards.map { (_, 1.0) })
    }.toMap
    
    new FiniteMarkovRewardProcess[S] {
      override def transitionRewardMap: RewardTransition[S] = rewardMap
    }
  }
  
  def getExperienceStream[S](
    fixedExperiences: Iterable[TransitionStep[S]]
  ): LazyList[TransitionStep[S]] = {
    val dist: Choose[TransitionStep[S]] = Choose(fixedExperiences)
    LazyList.continually(dist.sample)
  }
  
  def tdPrediction[S](
    experienceStream: Iterable[TransitionStep[S]],
    gamma: Double,
    numExperiences: Int
  ): Map[NonTerminal[S], Double] = {
    
    val valueApproximation: ValueFunctionApproximation[S] = TemporalDifference
      .tdPrediction(experienceStream, Tabular[NonTerminal[S]](), gamma)
      .drop(numExperiences)
      .next()
    
    valueApproximation.asInstanceOf[Tabular[NonTerminal[S]]].valuesMap
  }
  
}

object ExperienceReplayApp extends App {
  
  val logger: Logger = Logger("ExperienceReplayApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val givenData: Seq[Seq[(Char, Double)]] = Seq(
    Seq(('A', 2.0), ('A', 6.0), ('B', 1.0), ('B', 2.0)),
    Seq(('A', 3.0), ('B', 2.0), ('A', 4.0), ('B', 2.0), ('B', 0.0)),
    Seq(('B', 3.0), ('B', 6.0), ('A', 1.0), ('B', 1.0)),
    Seq(('A', 0.0), ('B', 2.0), ('A', 4.0), ('B', 4.0), ('B', 2.0), ('B', 3.0)),
    Seq(('B', 8.0), ('B', 2.0))
  )
  
  val gamma: Double = 0.9
  val numMCEpisodes: Int = 100000
  val numTDExperiences: Int = 1000000
  
  val fixedEpisodes: Iterable[Seq[TransitionStep[Char]]] = getFixedEpisodesFromStateRewards(
    stateRewardTraces = givenData,
    terminalState = 'T'
  )
  
  val returnSteps: Iterable[ReturnStep[Char]] = getReturnStepsFromFixedEpisodes(fixedEpisodes, gamma)
  
  val meanReturns: Map[NonTerminal[Char], Double] = getMeanReturnsFromReturnSteps(returnSteps)
  val meanReturnsStr: String = DynamicProgramming.valueFunctionToString(meanReturns)
  logger.info(f"Mean Returns:\n$meanReturnsStr")
  
  val episodes: LazyList[Seq[TransitionStep[Char]]] = getEpisodeStream(fixedEpisodes)
  val mcValueFunction: Map[NonTerminal[Char], Double] = mcPrediction(episodes, gamma, numMCEpisodes)
  val mcValueFunctionStr: String = DynamicProgramming.valueFunctionToString(mcValueFunction)
  logger.info(f"MC Value Function:\n$mcValueFunctionStr")
  
  val fixedExperiences: Iterable[TransitionStep[Char]] = fixedExperiencesFromFixedEpisodes(fixedEpisodes)
  
  val finiteMRP: FiniteMarkovRewardProcess[Char] = ExperienceReplayUtils.finiteMRP(fixedExperiences)
  val valueFunction = finiteMRP.valueFunctionVector(gamma)
  val valueFunctionStr = finiteMRP.valueFunctionToString(gamma)
  
  logger.info(f"Implied-MRP Value Function:\n$valueFunctionStr")
  
  val experiences = getExperienceStream(fixedExperiences)
  val tdValueFunction: Map[NonTerminal[Char], Double] = tdPrediction(experiences, gamma, numTDExperiences)
  val tdValueFunctionStr: String = DynamicProgramming.valueFunctionToString(tdValueFunction)
  logger.info(f"TD Value Function:\n$tdValueFunctionStr")
  
}
