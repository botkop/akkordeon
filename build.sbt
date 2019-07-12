name := "akkordeon"

version := "1.0"

scalaVersion := "2.12.6"

lazy val akkaVersion = "2.5.18"

libraryDependencies ++= Seq(
  "com.typesafe.akka" %% "akka-actor" % akkaVersion,
  "com.typesafe.akka" %% "akka-testkit" % akkaVersion,

  "com.typesafe.akka" %% "akka-remote" % akkaVersion,
  "com.typesafe.akka" %% "akka-cluster" % akkaVersion,
  "com.typesafe.akka" %% "akka-cluster-metrics" % akkaVersion,
  "com.typesafe.akka" %% "akka-cluster-tools" % akkaVersion,

  "com.github.romix.akka" %% "akka-kryo-serialization" % "0.5.1",

  "io.kamon" % "sigar-loader" % "1.6.6-rev002",
  
  "be.botkop" %% "scorch" % "0.1.1",
  "org.nd4j" % "nd4j-kryo_2.11" % "0.9.1",

  "com.typesafe.akka" %% "akka-slf4j" % akkaVersion,
  "ch.qos.logback" % "logback-classic" % "1.2.3",

  "org.scalatest" %% "scalatest" % "3.0.5" % "test"
)

// when OMP_NUM_THREADS is set to default by nd4j, 
// then lots more CPU is used for no notable performance gain
// set setting it to 1
envVars := Map("OMP_NUM_THREADS" -> "1")

fork := true
