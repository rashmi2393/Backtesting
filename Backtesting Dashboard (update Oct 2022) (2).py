# Databricks notebook source
# MAGIC %md
# MAGIC # Backtesting Dashboard
# MAGIC ### Input Descriptions:
# MAGIC ##### 1.Start Date: Earlier date of interest, format YYYY-MM-DD 
# MAGIC ##### 2. End Date: Later date of interest, format YYYY-MM-DD
# MAGIC ##### 3. order ID: Order ID number, can be found from GAM. Comma separated list if more than 1.
# MAGIC ##### 4. Model 1 TID: TID of the first model
# MAGIC ##### 5. Model 2 TID: TID of the second model 
# MAGIC ##### 6. Line Item IDs: OPTIONAL. Comma separated list if more than 1.

# COMMAND ----------

#setup
from pyspark.sql import functions as F 
from pyspark.sql.types import *
from datetime import datetime, timedelta, date

#plotting
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tkr
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

# COMMAND ----------

#read in user input data from widgets.

#1.Start Date: Earlier date of interest, format YYYY-MM-DD
dbutils.widgets.text("startDate", "", "1. Start Date")
start_date = dbutils.widgets.get('startDate')

#2. End Date: Later date of interest, format YYYY-MM-DD
dbutils.widgets.text("endDate", "", "2. End Date")
end_date = dbutils.widgets.get('endDate')

#3. order ID: Order ID number, can be found from GAM
dbutils.widgets.text("orderID", "", "3. order ID")
order_ID = dbutils.widgets.get('orderID')
#convert to list
order_ID = [int(x) for x in order_ID.split(',')]

#4. Model 1 TID: TID of the first model
dbutils.widgets.text("model1TID", "", "4. Model 1 TID")
model1_TID = dbutils.widgets.get('model1TID')

#5. Model 2 TID: TID of the second model
dbutils.widgets.text("model2TID", "", "5. Model 2 TID")
model2_TID = dbutils.widgets.get('model2TID')

#6. Line Item IDs
dbutils.widgets.text("LineItemIDs", "", "6. Line Item IDs")
line_item_ids = dbutils.widgets.get('LineItemIDs')

if line_item_ids:
  #convert to list
  line_item_ids = [int(x) for x in line_item_ids.split(',')]

#Collect the full model names for labeling plots and sections. Example: 'Model1: Spire > Fashion > Interest > Clothes > All > Luxury Retail'
model1_name = spark.read.table('segment_ref_prod.segment_trait').filter(F.col('trait_id') == model1_TID).select('segment_name').collect()[0][0]
model2_name = spark.read.table('segment_ref_prod.segment_trait').filter(F.col('trait_id') == model2_TID).select('segment_name').collect()[0][0]

# COMMAND ----------

#Show which orders are being analyzed

gam_li = spark.read.table('gam_prod.line_items')
orders_text = []
for orders in order_ID:
  order_current = gam_li.filter(F.col('orderId')== orders).select('orderName').orderBy('creationDateTime').collect()[0][0]
  orders_text.append(order_current)

order_name_table = spark.createDataFrame(orders_text, "string").toDF("Orders_Analyzed")
display(order_name_table)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Decile Groupings:
# MAGIC 1. H = Deciles 1-3 (High propensity)
# MAGIC 2. M = Deciles 4-7 (Medium propensity)
# MAGIC 3. L = Deciles 8-10 (Low propensity)

# COMMAND ----------

#Match clicks and impressions for cases where xid is not null for people in order_ID between dates start_date,end_date.

def match_clicks_impressions(order_ID,start_date,end_date):

  #Clicks from GAM
  if line_item_ids:
    gam_clks = (spark.read.table('gam_prod.networkclicks').filter(F.col('OrderId').isin(order_ID))\
              .filter(F.col("logdate").between(start_date,end_date)).\
              withColumn('xid',F.col('CustomTargeting')['vnd_4d_xid'])\
              .filter(F.col('xid').isNotNull()).filter(F.col('LineItemId').isin(line_item_ids)))\
    .withColumn('clicks',F.lit(1))

    #Impressions from GAM
    gam_imps = (spark.read.table('gam_prod.networkimpressions')\
              .filter(F.col('OrderId').isin(order_ID))\
              .filter(F.col("logdate").between(start_date,end_date))\
              .withColumn('xid',F.col('CustomTargeting')['vnd_4d_xid'])\
              .filter(F.col('xid').isNotNull()).filter(F.col('LineItemId').isin(line_item_ids)))\
    .withColumn('impressions',F.lit(1))\
              .withColumnRenamed("logdate", 'date')\
              .withColumnRenamed("xid", 'xid_imp')
    
      
  else:
    gam_clks = (spark.read.table('gam_prod.networkclicks').filter(F.col('OrderId').isin(order_ID))\
              .filter(F.col("logdate").between(start_date,end_date)).\
              withColumn('xid',F.col('CustomTargeting')['vnd_4d_xid'])\
              .filter(F.col('xid').isNotNull()))\
              .withColumn('clicks',F.lit(1))
                                                        
    gam_imps = (spark.read.table('gam_prod.networkimpressions')\
              .filter(F.col('OrderId').isin(order_ID))\
              .filter(F.col("logdate").between(start_date,end_date))\
              .withColumn('xid',F.col('CustomTargeting')['vnd_4d_xid'])\
              .filter(F.col('xid').isNotNull()))\
              .withColumn('impressions',F.lit(1))\
              .withColumnRenamed("logdate", 'date')\
              .withColumnRenamed("xid", 'xid_imp')

      
      
  cond = [gam_imps.Time == gam_clks.Time,
                 gam_imps.xid_imp == gam_clks.xid,
                 gam_imps.AdvertiserId == gam_clks.AdvertiserId,
                 gam_imps.OrderId == gam_clks.OrderId,
                 gam_imps.LineItemId == gam_clks.LineItemId,
                 gam_imps.CreativeId == gam_clks.CreativeId,
                 gam_imps.CreativeVersion == gam_clks.CreativeVersion,
                 gam_imps.CreativeSize == gam_clks.CreativeSize,
                 gam_imps.AdUnitId == gam_clks.AdUnitId,
                 gam_imps.CityId == gam_clks.CityId,
                 gam_imps.TimeUsec == gam_clks.TimeUsec,
                 gam_imps.Product == gam_clks.Product]


  clicks_and_impressions = ((gam_imps.join(gam_clks, cond, how="left")).fillna(0)).groupBy('xid_imp','date').agg(F.sum('impressions').alias('impressions'),F.sum('clicks').alias('clicks'))
  #gam_imps.agg(F.sum('impressions').alias('impressions')).show()
  #Join Clicks and Impressions to XID (left outer join)
  #clicks_and_impressions = gam_imps.join(gam_clks, 'xid', 'left_outer')\
  #.fillna(0).cache()

  return clicks_and_impressions

clicks_and_impressions = match_clicks_impressions(order_ID,start_date,end_date)

# COMMAND ----------

#Create model 1 section label
line_m1 = 'Model 1: ' + str(model1_name)

displayHTML("<table style='border-collapse:collapse;width:100%' border='2',>\
<tr><td style='padding:15px;font-size: 15pt'>{0}</td></tr>\
</table>".format(line_m1, "red"))

# COMMAND ----------

# Match impressions and clicks to model deciles.

def match_ctr_by_decile(start_date,end_date,modelTID,clicks_and_impressions):

  # Spire scores, read in for date and tid only.
  scores = spark.read.table('spire_prod.scores')
  scores_filtered_duplicates = scores.filter(F.col("date").between(start_date,end_date))\
  .filter(F.col("tid") == modelTID)
  
  scores_filtered = scores_filtered_duplicates.dropDuplicates()
  
  
  
  #display(scores)
  #display(scores_filtered)
  #Left outer join scores (deciles) to users and calculate ctr
  
  cond = [clicks_and_impressions.date == scores_filtered.date,
          clicks_and_impressions.xid_imp == scores_filtered.xid]

  ctr_by_decile = clicks_and_impressions.join(scores_filtered.filter(F.col('tid')== modelTID), cond, how ='left_outer')\
    .groupBy('tid','decile').agg(F.sum('impressions').alias('impressions'), F.sum('clicks').alias('clicks')).orderBy('decile')\
  .withColumn('ctr', F.round((F.col('clicks')/F.col('impressions')*100),2))
  
 #display(ctr_by_decile)

  return ctr_by_decile

model1_ctr_by_decile =  match_ctr_by_decile(start_date,end_date,model1_TID,clicks_and_impressions)

#Display table. tid is dropped for dashboard readability.
display(model1_ctr_by_decile.select(F.col('decile'), F.format_number(F.col('impressions'), 0).alias('impressions') , F.format_number(F.col('clicks'), 0).alias('clicks'), F.col('ctr').alias('ctr %')))
#display(scores_filtered)


# COMMAND ----------

#reshape ctr to np array for heatmap. Assumes order is maintained from previous steps
model1_ctr_by_decile_reshaped =  np.array(model1_ctr_by_decile.select(F.col("ctr")).collect()).reshape(11,1)

#remove null 
model1_ctr_by_decile_reshaped = np.delete(model1_ctr_by_decile_reshaped, 0, 0)

# COMMAND ----------

 #create a heatmap of ctr by decile. Takes the reshaped np array and labels for the y axis.
 def create_heatmap_tid_deciles(input_model, y_axis_labels):
  
   #font size
   sns.set(font_scale=1.6)
  
   #create heatmap
   output_heatmap = sns.heatmap(input_model, annot=True, cbar=True, cmap="Greens", yticklabels= y_axis_labels, fmt='.2g')
  
   for t in output_heatmap.texts: t.set_text(t.get_text() + " %")

   #move the x-axis label to the top for readability
   output_heatmap.xaxis.tick_top() # x axis on top
   output_heatmap.tick_params(length=0) #remove ticks from top
   output_heatmap.xaxis.set_label_position('top') 
  
   #label 
   output_heatmap.set(xlabel = 'ctr', ylabel='Decile', xticklabels=[])

#create the heatmap for model 1 deciles
create_heatmap_tid_deciles(model1_ctr_by_decile_reshaped, ['1','2','3','4','5','6','7', '8', '9', '10'])


# COMMAND ----------

# Reaggregate deciles into Low, Medium, High. Low: 1,2,3 Medium: 4,5,6,7 High: 8,9,10
def group_ctr_decile_groups(ctr_by_decile):
  
  #redistribute deciles to decile groupings
  LMH_deciles_matched = ctr_by_decile.withColumn("DecileGroup",F.when(F.col('decile') <=3, "1. H")\
                                                 .when(F.col('decile').between(4,7), "2. M")\
                                                 .when(F.col('decile') >= 8, "3. L")\
                                                .otherwise("Unscored"))
  #aggregate clicks and impressions and calculate ctr
  LMH_deciles_grouped = LMH_deciles_matched.groupBy('tid','DecileGroup').agg(F.sum('impressions').alias('impressions'),\
                                                                             F.sum('clicks').alias('clicks')).sort(F.col('tid').asc(), F.col('DecileGroup').asc()).orderBy('DecileGroup').withColumn('ctr', F.round((F.col('clicks')/F.col('impressions')*100),2))

#display(LMH_deciles_matched)
  return LMH_deciles_grouped


model1_LMH_deciles_grouped = group_ctr_decile_groups(model1_ctr_by_decile)

#Display table. tid is dropped for dashboard readability.
display(model1_LMH_deciles_grouped.select(F.col('DecileGroup'), F.format_number(F.col('impressions'), 0).alias('impressions') , F.format_number(F.col('clicks'), 0).alias('clicks'), F.col('ctr').alias('ctr %')))

# COMMAND ----------

#reshape ctr to np array for heatmap. Assumes order is maintained from previous steps
model1_ctr_by_deciles_grouped_reshaped =  np.array(model1_LMH_deciles_grouped.select("ctr").collect()).reshape(4,1)

#remove null 
model1_ctr_by_deciles_grouped_reshaped = np.delete(model1_ctr_by_deciles_grouped_reshaped, 3, 0)

# COMMAND ----------

 #create a heatmap of ctr by decile grouping. Takes the reshaped np array and labels for the y axis.
 def create_heatmap_tid_deciles_grouped(input_model, y_axis_labels):
  
   #font size
   sns.set(font_scale=1.6)
  
   #create heatmap
   output_heatmap = sns.heatmap(input_model, annot=True, cbar=True, cmap="Greens", yticklabels= y_axis_labels, fmt='.2g')
   
   for t in output_heatmap.texts: t.set_text(t.get_text() + " %")

   #move the x-axis label to the top for readability
   output_heatmap.xaxis.tick_top() # x axis on top
   output_heatmap.tick_params(length=0) #remove ticks from top
   output_heatmap.xaxis.set_label_position('top') 
  
   #labels 
   output_heatmap.set(xlabel = 'ctr', ylabel='Decile', xticklabels=[])

#create the heatmap for model 1 deciles grouped
create_heatmap_tid_deciles_grouped(model1_ctr_by_deciles_grouped_reshaped, ['1. H', '2. M', '3. L'])


# COMMAND ----------

#Create model 2 section label
line_m2 = 'Model 2: ' + str(model2_name)

displayHTML("<table style='border-collapse:collapse;width:100%' border='2',>\
<tr><td style='padding:15px;font-size: 15pt'>{0}</td></tr>\
</table>".format(line_m2, "red"))

# COMMAND ----------

#repeat process done on model 1 for model 2

#match by decile and decile grouping
model2_ctr_by_decile =  match_ctr_by_decile(start_date,end_date,model2_TID,clicks_and_impressions)
model2_LMH_deciles_grouped = group_ctr_decile_groups(model2_ctr_by_decile)

#display ctr tables for decile and decile grouping. tid is dropped for readability.
display(model2_ctr_by_decile.select(F.col('decile'), F.format_number(F.col('impressions'), 0).alias('impressions') , F.format_number(F.col('clicks'), 0).alias('clicks'), F.col('ctr').alias('ctr %')))
display(model2_LMH_deciles_grouped.select(F.col('DecileGroup'), F.format_number(F.col('impressions'), 0).alias('impressions') , F.format_number(F.col('clicks'), 0).alias('clicks'), F.col('ctr').alias('ctr %')))

# COMMAND ----------

#reshape ctr to np array for heatmap. Assumes order is maintained from previous steps
model2_ctr_by_decile_reshaped =  np.array(model2_ctr_by_decile.select("ctr").collect()).reshape(11,1)

#remove null 
model2_ctr_by_decile_reshaped = np.delete(model2_ctr_by_decile_reshaped, 0, 0)


# COMMAND ----------

#create the decile heatmap
create_heatmap_tid_deciles(model2_ctr_by_decile_reshaped, ['1','2','3','4','5','6','7', '8', '9', '10'])

# COMMAND ----------

#reshape ctr to np array for heatmap. Assumes order is maintained from previous steps
model2_ctr_by_deciles_grouped_reshaped =  np.array(model2_LMH_deciles_grouped.select("ctr").collect()).reshape(4,1)

# COMMAND ----------

#drop null
model2_ctr_by_deciles_grouped_reshaped = np.delete(model2_ctr_by_deciles_grouped_reshaped, 3, 0)

# COMMAND ----------

#create the decile grouping heatmap
create_heatmap_tid_deciles_grouped(model2_ctr_by_deciles_grouped_reshaped, ['1. H', '2. M', '3. L'])

# COMMAND ----------

# MAGIC %md
# MAGIC #Significance test
# MAGIC Are the H decile groupings of the two models significantly different?
# MAGIC [Source](https://condenast-dev.cloud.databricks.com/?o=1862559208603375#notebook/4061107375989658/command/4061107375989682)

# COMMAND ----------

#collect counts and nobs for proportions_ztest
counts1 = model1_LMH_deciles_grouped.filter(F.col("DecileGroup") == '1. H').select("clicks").collect()[0][0]
#print(counts1)

counts2 = model2_LMH_deciles_grouped.filter(F.col("DecileGroup") == '1. H').select("clicks").collect()[0][0]
#print(counts2)

nobs1 = model1_LMH_deciles_grouped.filter(F.col("DecileGroup") == '1. H').select("impressions").collect()[0][0]
#print(nobs1)

nobs2 = model2_LMH_deciles_grouped.filter(F.col("DecileGroup") == '1. H').select("impressions").collect()[0][0]
#print(nobs2)

# COMMAND ----------

#calculate the p-value using proportions_ztest
def calculate_p_value(counts1, counts2, nobs1, nobs2):
    counts = [counts1,counts2] #clicks
    nobs = [nobs1,nobs2] #impressions
    zstat, p_value = proportions_ztest(counts,nobs)
    return float(p_value) 

statsPValue = calculate_p_value(counts1, counts2, nobs1, nobs2)

# COMMAND ----------

#display the results from proportions_ztest to a table. 
line1 = "Statistical Test Results:"

alpha = 0.05 #compare to p_value
line2 = "The calculated p_value is: " + str(statsPValue)
line3 = "alpha is: " + str(alpha)

if statsPValue < alpha:
  line4 = "The p_value is less than alpha. The null hypothesis is rejected, and the models are significantly different."
else:
  line4 = "The p_value is greater than or equal to alpha. The null hypothesis is NOT rejected, and the models are NOT significantly different."
    
displayHTML("<table style='border-collapse:collapse;width:100%' border='2',>\
<tr><td style='padding:15px;font-size: 25pt'>{0}</td></tr>\
<tr><td style='padding:15px;color:{4}'>{1}</td></tr>\
<tr><td style='padding:15px;color:{5}'>{2}</td></tr>\
<tr><td style='padding:15px;color:{6}'>{3}</td></tr>\
</table>".format(line1, line2, line3, line4, "black", "black", "red"))


# COMMAND ----------

# MAGIC %md
# MAGIC # Overlap:
# MAGIC Are the same users targeted in each decile / decile grouping for the two models? For all the times a user is seen, the maxium score (lowest decile) is chosen to represent the score of the user during the date range.
# MAGIC 
# MAGIC # Important Note:
# MAGIC The overlap part is for  “scored users” (i.e. not exposed - this part isn’t tied to the above campaign results)

# COMMAND ----------

#compare the overlap of the two models by comparing the lowest deciles the users were assigned to in the date range.
#NOTE: For all the times a user is seen, the maxium score (lowest decile) is chosen to represent the score of the user during the date range. This could be changed to be an average, highest frequency, or some other assessment in the future.

def compare_two_models(model1_TID,model2_TID,start_date,end_date):
  
  #read in scores for each model. For all the times a user is seen, the maxium score (lowest decile) is chosen to represent the score of the user during the date range. This could be changed to be an average or some other assessment in the future.
  scores = spark.read.table('spire_prod.scores').filter(F.col("date").between(start_date,end_date))
  tid_1_Scores = scores.filter(F.col('tid') == model1_TID).groupby('xid').agg(F.min('decile').alias('tid1Decile'),\
                                                                      F.max('score').alias('tid1Score'))
  tid_2_Scores = scores.filter(F.col('tid') == model2_TID).groupby('xid').agg(F.min('decile').alias('tid2Decile'),\
                                                                      F.max('score').alias('tid2Score'))
  
  #count how many users per decile overlap (by xid) for each of the 2 segments
  model_overlap = tid_1_Scores.join(tid_2_Scores,'xid').groupBy('tid1Decile','tid2Decile').count().orderBy('tid1Decile','tid2Decile')
  
  overlap2 = tid_1_Scores.join(tid_2_Scores,'xid')
  #Display the correlation coefficient
  displayHTML("<table style='border-collapse:collapse;width:100%' border='2',>\
<tr><td style='padding:15px;font-size: 15pt'>{0}</td></tr>\
</table>".format('The correlation between the scores of the two models is: ' + str(overlap2.stat.corr('tid1Score','tid2Score')), "red"))
  
  
  
  # Reaggregate deciles into Low, Medium, High. Low: 1,2,3 Medium: 4,5,6,7 High: 8,9,10
  tid1LMH = tid_1_Scores.withColumn("DecileGroup1",F.when(F.col('tid1Decile') <=3, "1. H")\
                                                 .when(F.col('tid1Decile').between(4,7), "2. M")\
                                                 .when(F.col('tid1Decile') >= 8, "3. L")\
                                                 .otherwise("Unscored"))

# Reaggregate deciles into Low, Medium, High. Low: 1,2,3 Medium: 4,5,6,7 High: 8,9,10
  tid2LMH = tid_2_Scores.withColumn("DecileGroup2",F.when(F.col('tid2Decile') <=3, "1. H")\
                                                 .when(F.col('tid2Decile').between(4,7), "2. M")\
                                                 .when(F.col('tid2Decile') >= 8, "3. L")\
                                                 .otherwise("Unscored"))

  #Join Decile Groups together to find overlap
  model_overlap_LMH = tid1LMH.join(tid2LMH,'xid').groupBy('DecileGroup1','DecileGroup2')\
  .count().orderBy('DecileGroup1','DecileGroup2')
  
  #display(tid_1_Scores)
  #display(tid_2_Scores)

  return model_overlap, model_overlap_LMH


model_overlap, model_overlap_LMH = compare_two_models(model1_TID,model2_TID,start_date,end_date)

#display(model_overlap)
#display(model_overlap_LMH)


#display(tid1LMH)
#display(tid2LMH)


# COMMAND ----------

# MAGIC %md
# MAGIC # Overlap:
# MAGIC 
# MAGIC Overlap by Deciles:

# COMMAND ----------

#reshape data into a 10 by 10 to prepare for heatmap for decile overlay
model_overlap_reshaped =  np.array(model_overlap.select("count").collect()).reshape(10,10)
#find percent composition by column
model_overlap_reshaped_percents = model_overlap_reshaped/model_overlap_reshaped.sum(axis=0)

# COMMAND ----------

#create heatmap for deciles overlay. 
def create_heatmap_deciles(input_model, fmt_param, cbar_param, commas):
 #set up figure 
  sns.set(font_scale=2.2)
  plt.figure(figsize=(40,15))

  #create heatmap
  output_heatmap = sns.heatmap(input_model, annot=True, fmt=fmt_param, cbar=True, cbar_kws=cbar_param, xticklabels=np.linspace(1,10,10, dtype = int), yticklabels=np.linspace(1,10,10, dtype = int), cmap="Greens")

  #move the x-axis label to the top for readability
  output_heatmap.xaxis.tick_top() # x axis on top
  output_heatmap.tick_params(length=0) #remove ticks from top
  output_heatmap.xaxis.set_label_position('top') 
  
  #add commas to the numbers for readability
  if commas:
    for t in output_heatmap.texts:
      t.set_text('{:,d}'.format(int(t.get_text())))
  
  #label 
  output_heatmap.set(xlabel='Model 2:  ' + str(model2_name), ylabel='Model 1:  '  + str(model1_name))

  #put a rectangular box on the diagonals
  for x in range (0,10):
    output_heatmap.add_patch(Rectangle((x,x), 1, 1, fill=False, edgecolor='red', lw=3))
    
  return output_heatmap

heatmap_model_overlap = create_heatmap_deciles(model_overlap_reshaped, 'n', {'format':tkr.ScalarFormatter(useMathText=True)}, True)

display(heatmap_model_overlap)
  

# COMMAND ----------

#show model overlap for percent composition of the columns. When added, columns equal 100% for Model 2. Shows what percent overlap there is per decile for the two models.
heatmap_model_overlap_percents = create_heatmap_deciles(model_overlap_reshaped_percents, '.0%', {'format': FuncFormatter(lambda x,pos: '{:.0%}'.format(x))}, False )

display(heatmap_model_overlap_percents)

# COMMAND ----------

# MAGIC %md
# MAGIC Overlap by Decile Groups:

# COMMAND ----------

#reshape data into a 3 by 3 to prepare for heatmap for decile overlay
model_overlap_LMH_reshaped =  np.array(model_overlap_LMH.select("count").collect()).reshape(3,3)
#find percent composition by column
model_overlap_LMH_reshaped_percents = model_overlap_LMH_reshaped/model_overlap_LMH_reshaped.sum(axis=0)

# COMMAND ----------

#create heatmap for deciles grouped overlay. 
def create_heatmap_LMH(input_model, fmt_param, cbar_param, commas):

  #set up figure 
  sns.set(font_scale=2.2)
  plt.figure(figsize=(40,15))

  #create heatmap
  output_heatmap = sns.heatmap(input_model, annot=True, fmt=fmt_param, cbar=True, cbar_kws=cbar_param, xticklabels=['1- H', '2- M', '3- L'], yticklabels=['1- H', '2- M', '3- L'], cmap="Greens")

  #move the x-axis label to the top for readability
  output_heatmap.xaxis.tick_top() # x axis on top
  output_heatmap.tick_params(length=0) #remove ticks from top
  output_heatmap.xaxis.set_label_position('top') 
  
  if commas:
    for t in output_heatmap.texts:
      t.set_text('{:,d}'.format(int(t.get_text())))
  
  #label 
  output_heatmap.set(xlabel='Model 2:  ' + str(model2_name), ylabel='Model 1:  '  + str(model1_name))

  #put a rectangular box on the diagonals
  for x in range (0,3):
    output_heatmap.add_patch(Rectangle((x,x), 1, 1, fill=False, edgecolor='red', lw=3))
    
  return output_heatmap

heatmap_model_overlap_LMH = create_heatmap_LMH(model_overlap_LMH_reshaped, 'n', {'format':tkr.ScalarFormatter(useMathText=True)}, True)

display(heatmap_model_overlap_LMH)
  

# COMMAND ----------

#show model overlap for percent composition of the columns. When added, columns equal 100% for Model 2. Shows what percent overlap there is per decile grouping for the two models.
heatmap_model_overlap_LMH_percents = create_heatmap_LMH(model_overlap_LMH_reshaped_percents, '.0%', {'format': FuncFormatter(lambda x,pos: '{:.0%}'.format(x))}, False )
display(heatmap_model_overlap_LMH_percents)

# COMMAND ----------


