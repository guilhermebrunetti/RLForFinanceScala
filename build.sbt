name := "RLForFinanceScala"

version := "0.1"

scalaVersion := "2.13.6"

resolvers += "Sonatype OSS Snapshots" at
  "https://oss.sonatype.org/content/repositories/snapshots"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "1.2",
  //Optional - the 'why' is explained in the How it works section
  "org.scalanlp" %% "breeze-natives" % "1.2",
  "org.scalanlp" %% "breeze-viz" % "1.2",
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "org.scalatest" %% "scalatest" % "3.0.8" % Test,
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.4",
  "org.slf4j" % "slf4j-api" % "1.7.32",
  "org.slf4j" % "slf4j-simple" % "1.7.32",
  // https://mvnrepository.com/artifact/com.storm-enroute/scalameter
  "com.storm-enroute" %% "scalameter" % "0.21" % Test,
  "com.storm-enroute" %% "scalameter-core" % "0.21",
  "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
)

testFrameworks += new TestFramework("org.scalameter.ScalaMeterFramework")

logBuffered := false