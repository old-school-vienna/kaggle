install.packages("tidyverse")
install.packages("ggplot2")

df_shops = aggregate(data.frame(df_sales3$item_cnt_month), by=data.frame(df_sales3$shop_id, df_sales3$date_block_num), sum, na.rm=TRUE)
colnames(df_shops) = c("shop_id", "month", "item_cnt_month")

df_shops_sum = aggregate(data.frame(df_shops$item_cnt_month), by=data.frame(df_shops$shop_id), sum)
colnames(df_shops_sum) = c("shop_id", "item_cnt_month")

df_shops_cnt = aggregate(data.frame(df_shops$item_cnt_month), by=data.frame(df_shops$shop_id), length)
colnames(df_shops_cnt) = c("shop_id", "item_cnt_month")

df_shops = merge(x = df_shops, y = df_shops_sum, by = "shop_id", all = TRUE)
df_shops = df_shops[with(df_shops, order(item_cnt_month.y)), ]


ggplot(df_shops, aes(x = reorder(shop_id, item_cnt_month.y), y = item_cnt_month.x)) +
  geom_point() +
  labs(title="sales of shops / sum of products", x='shop_id') 

