package rl.externalExamples

import breeze.plot.{Figure, plot}
import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming._
import rl.FiniteMarkovDecisionProcess.{StateActionMapping, processInputMap}
import rl.utils.Categorical
import rl.{FiniteDeterministicPolicy, FiniteMarkovDecisionProcess, NonTerminal}

import java.time.LocalDateTime
import java.util.Locale

/** *
 * Modified problem based on the previous GamblersProblemMDP.
 * "
 * A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
 * If the coin comes up heads, he wins as many dollars as he has staked on that flip;
 * if it is tails, he loses his stake.
 * "
 *
 * The game ends when the gambler obtains at least $100, or loses by running out of money.
 * On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars.
 * On each flip, the instantaneous reward is the stake he/she bet.
 *
 * The original Gambler's problem was designed to find the probability of winning the game at terminal state.
 * In this variation, our value function should be the expected gain from betting.
 *
 * If the probability of heads is less than 0.5, the optimal policy is to never play.
 *
 * Interesting, if the probability of heads is less than 0.6, the optimal policy is to bet 1 coin
 * at every flip, until we reach the almost-final state, and then bet a bigger amount.
 *
 * If the probability is equal to 0.5, then in expected values of each bet is 0 and the gambler
 * is indifferent between betting or not betting. There are many optimal policies in this scenario.
 *
 */
class GamblersProblemMDPv2(
  val goal: Int,
  val probability: Double,
) extends FiniteMarkovDecisionProcess[Int, Int] {
  require(
    probability > 0.0 && probability < 1.0,
    s"Probability should be a number strictly between 0 and 1; got instead $probability"
  )

  require(goal > 0, s"Goal should be a strictly positive integer numbers; got instead $goal")

  override val stateActionMap: StateActionMapping[Int, Int] = {
    val inputMap = (1 until goal).map { i =>
      i -> (0 to i).map { a =>
        a -> Categorical(
          Map(
            (i + a, a.toDouble) -> probability,
            (i - a, -a.toDouble) -> (1.0 - probability)
          )
        )
      }.toMap
    }.toMap

    processInputMap(inputMap)
  }

  override def stateSortingFunction(x: NonTerminal[Int], y: NonTerminal[Int]): Boolean = x.state <= y.state

  override def actionSortingFunction(x: Int, y: Int): Boolean = x > y
}

object GamblersProblemMDPv2 {
  def apply(goal: Int, probability: Double): GamblersProblemMDPv2 =
    new GamblersProblemMDPv2(goal, probability)

}

object GamblersProblemMDPv2App extends App {

  val logger: Logger = Logger("GamblersProblemMDPv2App")
  Locale.setDefault(Locale.US) // To print numbers in US format

  val goal: Int = 10
  val probability: Double = 0.51
  val gamblersProblemMDPv2 = GamblersProblemMDPv2(goal, probability)
  val sortingFunction: (NonTerminal[Int], NonTerminal[Int]) => Boolean = gamblersProblemMDPv2.stateSortingFunction

  val gamma = 1.0

  val nonTerminalStates = gamblersProblemMDPv2.nonTerminalStates

  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val (optimalValueFunction: ValueFunction[Int], optimalPolicy: FiniteDeterministicPolicy[Int, Int]) =
    valueIterationResult(gamblersProblemMDPv2, gamma)
  logger.info(s"Finished computation at ${LocalDateTime.now()}")

  // Policy of making of always betting all the money the agent has
  val actionMap = (1 until goal).map(i => i -> i).toMap
  val maxBetPolicy = FiniteDeterministicPolicy(actionMap = actionMap)

  val impliedMRP = gamblersProblemMDPv2.applyFinitePolicy(maxBetPolicy)
  val impliedValueFunction = impliedMRP.valueFunctionVector(gamma)
  val impliedValueFunctionStr = impliedMRP.valueFunctionToString(gamma)

  val optimalVfStr = valueFunctionToString(optimalValueFunction, Some(sortingFunction))

  val xs = nonTerminalStates.map(_.state.toDouble)
  val optimalValues = nonTerminalStates.map(optimalValueFunction.apply)
  val optimalActions = nonTerminalStates.map(nt => optimalPolicy.actionForState(nt.state).toDouble)
  val maxBetActions = nonTerminalStates.map(nt => maxBetPolicy.actionForState(nt.state).toDouble)

  val fig = Figure("Gambler's Problem v2 Optimal Value Function and Max-Bet Policy Value Function")
  val p = fig.subplot(0)
  p += plot(xs, optimalValues, name = "Optimal Value Function")
  p += plot(xs, impliedValueFunction, name = "Max-Bet Policy Value Function")
  p.legend = true
  p.setYAxisDecimalTickUnits()

  val fig2 = Figure("Gambler's Problem v2 Optimal Policy and Max Bet Policy")
  val p2 = fig2.subplot(0)
  p2 += plot(xs, optimalActions, style = '.', name = "Optimal Policy")
  p2 += plot(xs, maxBetActions, style = '+', name = "Max Bet Policy")
  p2.legend = true
  p2.setYAxisIntegerTickUnits()

  logger.info(f"Optimal Value Function:\n$optimalVfStr")
  logger.info(f"Optimal Policy:\n${optimalPolicy.printPolicy(_ < _)}")
  logger.info(f"Implied Value Function From Policy:\n$impliedValueFunctionStr")

  logger.info("Finishing App execution")

}

