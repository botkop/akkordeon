name := "akkordeon"

version := "1.0"

// scalaVersion := "2.12.6"
scalaVersion := "2.11.12"

lazy val akkaVersion = "2.5.16"

libraryDependencies ++= Seq(
  "com.typesafe.akka" %% "akka-actor" % akkaVersion,
  "com.typesafe.akka" %% "akka-testkit" % akkaVersion,
  "be.botkop" % "scorch_2.12" % "0.1.1-SNAPSHOT",
  "org.scalatest" %% "scalatest" % "3.0.5" % "test"
)

libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.3" % "0.7.1"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.3.1"

scalacOptions in ThisBuild ++= Seq("-feature")
fork := true
