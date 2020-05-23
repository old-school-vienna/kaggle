import java.nio.file.{Path, Paths}

import org.apache.spark.sql.SparkSession

object CreateSchema {

  def main(args: Array[String]): Unit = {
    val fileNam = "Sales5_Ab2011_InklPred.csv"

    val spark = SparkSession.builder.appName("Simple Application")
      .config("spark.master", "local")
      .getOrCreate()
    val datadir = System.getenv("DATADIR")
    if (datadir == null) throw new IllegalArgumentException("Environment variable DATADIR must be defined")
    val fp = Paths.get(datadir, fileNam)

    val df = spark.read.option("header", "true").csv(fp.toString)

    val row = df.first()
    val indent1 = "      "
    val indent2 = "            "
    println()
    println(s"${indent1}schema = T.StructType([")
    val fields = for (f <- row.schema) yield {
      val strNullable = if (f.nullable) "True" else "False"
      s"${indent2}T.StructField('${f.name}', T.${f.dataType}(), $strNullable)"
    }
    println(fields.mkString(",\n"))
    println(s"$indent1])")
    println()

  }

}
