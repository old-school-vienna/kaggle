
import java.nio.file.Path

import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.{Logger, LoggerFactory}


case class S501 (
                  year:Option[Int],
                  month:Option[Int],
                  dn:Option[Int],
                  wday:Option[Int],
                  snap:Option[Int],
                  dept_id:Option[String],
                  item_id:Option[String],
                  store_id:Option[String],
                  sales:Option[Double],
                  flag_ram:Option[Int],
                  Sales_Pred:Option[Double],
                  label:Option[Double],
                )



object Train {

  val log: Logger = LoggerFactory.getLogger("Train")

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Simple Application")
      .config("spark.master", "local")
      .getOrCreate()

    //print(createCaseClass(spark))
    trainit(spark)
  }
  private def trainit(spark: SparkSession): Unit = {
    import spark.implicits._
    log.info("-- Started train")
    val fnam = Path.of(System.getenv("DATADIR"), "s5_01_small.parquet")
    val ds: Dataset[S501] = spark.read.parquet(fnam.toString).as[S501]
    ds.show(20)
  }

  def createCaseClass(spark: SparkSession): String = {
    log.info("-- Started train")
    val fnam = Path.of(System.getenv("DATADIR"), "s5_01_small.parquet")
    implicit val conv: Schema2CaseClass = new Schema2CaseClass()
    val df = spark.read.parquet(fnam.toString)
    conv.schemaToCaseClass(df.schema, "S501")

  }

}
