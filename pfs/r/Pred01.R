library(reshape2)

set.seed(12345)

Pathcsv = "/data/kaggle/pfs/" # ww
# Pathcsv = "C://Users//karlb//Documents//SASUniversityEdition//myfolders//Predict Future Sales//"
# Pathcsv = "C://Users//Win10//Documents//SASUniversityEdition//myfolders//Predict Future Sales//" # Intel NUK

csvfile = paste(Pathcsv, "sales_train.csv", sep="")
df_sales = read.csv2(csvfile, sep=",", header=TRUE, dec=".", stringsAsFactors=TRUE)

csvfile = paste(Pathcsv, "items.csv", sep="")
df_items = read.csv2(csvfile, sep=",", header=TRUE, dec=".", stringsAsFactors=TRUE)

#csvfile = paste(Pathcsv, "item_categories.csv", sep="")
#df_itemcats = read.csv2(csvfile, sep=",", header=TRUE, dec=".", stringsAsFactors=TRUE)

#csvfile = paste(Pathcsv, "shops.csv", sep="")
#df_shops = read.csv2(csvfile, sep=",", header=TRUE, dec=".", stringsAsFactors=TRUE)

csvfile = paste(Pathcsv, "sample_submission.csv", sep="")
df_submit = read.csv2(csvfile, sep=",", header=TRUE, dec=".", stringsAsFactors=TRUE)

csvfile = paste(Pathcsv, "test.csv", sep="")
df_test = read.csv2(csvfile, sep=",", header=TRUE, dec=".", stringsAsFactors=TRUE)

############## Additional Features ################
df_sales2 = merge(df_sales, df_items[2:3], by="item_id")
df_sales2$year  = substring(as.character(df_sales2$date), 7, 10)
df_sales2$month = paste(substring(as.character(df_sales2$date), 7, 10), substring(as.character(df_sales2$date), 4, 5), sep="-")

############## Analyse #############
test_items = unique(df_test$item_id)
test_shops = unique(df_test$shop_id)
test_shopitems = unique(paste(as.character(df_test$shop_id), as.character(df_test$item_id), sep="-"))

length(unique(df_sales2$item_id))
length(unique(df_test$item_id))
length(unique(df_sales2$shop_id))
length(unique(df_test$shop_id))

############## Aggragate ############
df_sales3 = aggregate(data.frame(df_sales2$item_cnt_day), by=data.frame(df_sales2$date_block_num, df_sales2$shop_id, df_sales2$item_id), sum, na.rm=TRUE)
colnames(df_sales3) = c("date_block_num", "shop_id", "item_id", "item_cnt_month")
df_sales3$shopitem_id = paste(as.character(df_sales3$shop_id), as.character(df_sales3$item_id), sep="-")

train_shopitems = unique(df_sales3$shopitem_id)
miss = train_shopitems - test_shopitems

inter = intersect(test_shopitems, train_shopitems)

# df_pred1 = aggregate(data.frame(df_sales3$item_cnt_month), by=data.frame(df_sales3$shop_id, df_sales3$item_id), mean, na.rm=TRUE)
df_pred1 = aggregate(data.frame(df_sales4$item_cnt_month), by=data.frame(df_sales4$shop_id, df_sales4$item_id), mean, na.rm=TRUE)
colnames(df_pred1) = c("shop_id", "item_id", "item_cnt_month")
df_pred1 = merge(df_pred1, df_items[2:3], by= "item_id")

df_pred1c = aggregate(data.frame(df_pred1$item_cnt_month), by=data.frame(df_pred1$shop_id, df_pred1$item_category_id), mean, na.rm=TRUE)
colnames(df_pred1c) = c("shop_id", "item_category_id", "item_cnt_month_icatmean")

df_pred1i = aggregate(data.frame(df_pred1$item_cnt_month), by=data.frame(df_pred1$item_id), mean, na.rm=TRUE)
colnames(df_pred1i) = c("item_id", "item_cnt_month_itemmean")

df_pred2 = merge(df_test, df_pred1, by=c("shop_id", "item_id"), all.x=TRUE)
df_pred2 = merge(df_pred2, df_pred1c, by=c("shop_id", "item_category_id"), all.x=TRUE)
df_pred3 = merge(df_pred2, df_pred1i, by=c("item_id"), all.x=TRUE)

sum(is.na(df_pred3$item_cnt_month))

df_pred3$item_cnt_month = ifelse(is.na(df_pred3$item_cnt_month), df_pred3$item_cnt_month_itemmean, df_pred3$item_cnt_month)
df_pred3$item_cnt_month = round(df_pred3$item_cnt_month, digits=1)

sum(is.na(df_pred3$item_cnt_month))
df_pred3[is.na(df_pred3)]=0


############# Submit ###################
subfile = paste(paste(Pathcsv, "Test2_MeanMonths", sep=""), ".csv", sep="")
write.csv(df_pred3[3:4], row.names=FALSE, file=subfile)

