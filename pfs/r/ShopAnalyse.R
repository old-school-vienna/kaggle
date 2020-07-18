install.packages("tidyverse")
install.packages("ggplot2")

df_shops = aggregate(data.frame(df_sales3$item_cnt_month), by=data.frame(df_sales3$shop_id, df_sales3$date_block_num), sum, na.rm=TRUE)
colnames(df_shops) = c("shop_id", "month", "item_cnt_month")

ggplot(df_shops, aes(x = shop_id, y = item_cnt_month)) +
  geom_point() +
  labs(title="sales of shops") 
