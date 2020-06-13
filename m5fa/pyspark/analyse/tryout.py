from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from pprint import pprint

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.sql import SparkSession, DataFrame

import helpers as hlp
from common import fnam


def no_idea():
    spark = SparkSession.builder \
        .appName(__name__) \
        .getOrCreate()

    sc: SparkContext = spark.sparkContext

    rdd_long = sc.parallelize(range(0, 10000000))

    short = rdd_long.take(20)
    print(f"type from take: {type(short)}")

    rdd_short = sc.parallelize(short) \
        .filter(lambda x: x % 2 == 0)

    print(f"len of short: {rdd_short.collect()}")


def dataclass_example():
    @dataclass
    class X:
        nam: str
        cnt: int

    x = X(nam='hallo', cnt=11)

    print(x)


def save_model():
    path = Path("target")
    if not path.exists():
        path.mkdir()
    nam = "gr1.sp"

    def save_load_hlp():
        SparkSession.builder \
            .appName("tryout") \
            .getOrCreate()
        m = LinearRegression(regParam=0.5, maxIter=10)
        pm = m.extractParamMap()
        pprint(pm)
        print()
        pprint(str(path))
        hlp.save_model(m, path, nam)

        m2 = hlp.load_model(path, nam)
        pm2 = m2.extractParamMap()
        pprint(pm2)

    def load_hlp():
        SparkSession.builder \
            .appName("tryout") \
            .getOrCreate()

        m2 = hlp.load_model(path, nam)
        pm2 = m2.extractParamMap()
        pprint(pm2)

    save_load_hlp()


def load_model():
    SparkSession.builder \
        .appName("tryout") \
        .getOrCreate()
    eid = "glrgi"
    fn = fnam(eid)
    m = hlp.load_model(hlp.get_datadir(), fn)
    print(f"----reading {fn}----------")
    pprint(m.extractParamMap())


def submission():
    testval = [
        [1, 'HOBBIES_1_001', 'CA_1', 2.2],
        [1, 'HOBBIES_1_001', 'CA_2', 0.0],
        [1, 'HOBBIES_1_001', 'TX_1', 2.0],
        [1, 'HOBBIES_1_002', 'CA_1', 2.0],
        [1, 'HOBBIES_1_002', 'CA_2', 0.0],
        [1, 'HOBBIES_1_002', 'TX_1', 2.0],
        [2, 'HOBBIES_1_001', 'CA_1', 2.0],
        [2, 'HOBBIES_1_001', 'CA_2', 0.0],
        [2, 'HOBBIES_1_001', 'TX_1', 2.0],
        [2, 'HOBBIES_1_001', 'TX_2', 1.0],
        [2, 'HOBBIES_1_002', 'CA_1', 0.0],
        [2, 'HOBBIES_1_002', 'CA_2', 0.0],
        [2, 'HOBBIES_1_002', 'TX_1', 2.0],
        [3, 'HOBBIES_1_001', 'CA_1', 2.0],
        [3, 'HOBBIES_1_001', 'CA_2', 0.0],
        [3, 'HOBBIES_1_001', 'TX_1', 2.0],
        [3, 'HOBBIES_1_002', 'CA_1', 2.2],
        [3, 'HOBBIES_1_002', 'CA_2', 0.3],
        [3, 'HOBBIES_1_002', 'TX_1', 2.4],
        [4, 'HOBBIES_1_001', 'CA_1', 2.0],
        [4, 'HOBBIES_1_001', 'CA_2', 0.5],
        [4, 'HOBBIES_1_001', 'TX_1', 2.1],
        [4, 'HOBBIES_1_001', 'TX_2', 1.1],
        [4, 'HOBBIES_1_002', 'CA_1', 0.1],
        [4, 'HOBBIES_1_002', 'CA_2', 0.2],
        [4, 'HOBBIES_1_002', 'TX_1', 2.2],
    ]
    spark = SparkSession.builder \
        .appName("tryout") \
        .getOrCreate()
    schema = T.StructType([
        T.StructField('dn', T.IntegerType(), True),
        T.StructField('item_id', T.StringType(), True),
        T.StructField('store_id', T.StringType(), True),
        T.StructField('prediction', T.DoubleType(), True),
    ])

    df = spark.createDataFrame(testval, schema=schema)
    dfp = df.groupBy('item_id', 'store_id').pivot("dn").sum('prediction')

    def rename_col(cn: str) -> str:
        if cn in ['item_id', 'store_id']:
            return cn
        else:
            return f"F{cn}"

    new_cols = [rename_col(c) for c in dfp.columns]

    dfp1 = dfp.toDF(*new_cols)
    dfp1.show()


def params_of_GBTRegressor():
    """
    def param_grid_gbtr(esti: Estimator) -> list:
        return ParamGridBuilder() \
            .addGrid(esti.maxBins, [16, 32, 64]) \
            .addGrid(esti.maxDepth, [3, 5, 10]) \
            .build()

    
{
 Param(parent='GBTRegressor_a1b083f1027e', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32,
    (esti.maxBins, [01, 32])
 Param(parent='GBTRegressor_a1b083f1027e', name='lossType', doc='Loss function which GBT tries to minimize (case-insensitive). Supported options: squared, absolute'): 'squared',
    (esti.lossType, ['absolute', 'squared'])
 Param(parent='GBTRegressor_a1b083f1027e', name='featureSubsetStrategy', 
    (esti.featureSubsetStrategy ,['auto', 'all'])

    doc="The number of features to consider for splits at each tree node. Supported 
    options: 
        'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), 
        'sqrt' for classification and 
        'onethird' for regression), 
        'all' (use all features), 
        'onethird' (use 1/3 of the features), 
        'sqrt' (use sqrt(number of features)), 
        'log2' (use log2(number of features)), 
        'n' (when n is in the range (0, 1.0], use n *   number of features. When n is in the range (1, number of features), use n features). default = 'auto'"): 'all',
 Param(parent='GBTRegressor_a1b083f1027e', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1,
 Param(parent='GBTRegressor_a1b083f1027e', name='subsamplingRate', doc='Fraction of the training data used for learning each decision tree, in range (0, 1].'): 1.0,
 Param(parent='GBTRegressor_a1b083f1027e', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
    (esti.maxDepth, [3, 5, 8])
 Param(parent='GBTRegressor_a1b083f1027e', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext.'): 10,
 Param(parent='GBTRegressor_a1b083f1027e', name='cacheNodeIds', d
    oc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. C
    aching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval.'): False}

 Param(parent='GBTRegressor_a1b083f1027e', name='stepSize', doc='Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator.'): 0.1,
 Param(parent='GBTRegressor_a1b083f1027e', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0,
 Param(parent='GBTRegressor_a1b083f1027e', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size.'): 256,
 Param(parent='GBTRegressor_a1b083f1027e', name='maxIter', doc='max number of iterations (>= 0).'): 20,

    """
    SparkSession.builder \
        .appName("tryout") \
        .getOrCreate()
    m = GBTRegressor()
    pm = m.extractParamMap()
    pprint(pm)


@dataclass
class TrainKey:
    item_id: str
    store_id: str


def store_multiple_trained_models():
    print("-- store_multiple_trained_models")
    spark = SparkSession.builder \
        .appName("tryout") \
        .getOrCreate()

    # Create small df if not exists
    # hlp.create_small_dataframe(spark)

    # Read data and filter for traing data
    pp: DataFrame = hlp.readFromDatadirParquet(spark, "s5_01_medium") \
        .where(F.col("label").isNotNull())

    # Create key column
    key_udf = F.udf(lambda a, b: f"{b}", T.StringType())
    pp1 = pp.withColumn('key', key_udf(pp.item_id, pp.store_id))

    # pp1.show()
    pp1.describe().show()

    # data ordered by key
    pp2 = pp1 \
        .sort('key')

    keys = chain(*pp1.select("key").distinct().orderBy('key').take(5))
    for k in keys:
        pp3 = pp2.filter(f"key = '{k}'")
        print(f"-- {k}: {pp3.count()}")


store_multiple_trained_models()
