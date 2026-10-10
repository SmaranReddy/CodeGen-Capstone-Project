"""Classify the 135 local Spider2-Lite examples by the relational query patterns they need.

Output: data/interim/spider2_local_query_patterns.jsonl (+ prints counts).
NOT a converter. Reads manifest / raw files read-only.

`basis`:
  gold_sql  - table list and construct hints extracted from the shipped gold SQL (24 ids), then hand-reviewed.
  inferred  - NO gold SQL exists (111 ids): tables/constructs are my reading of question + schema, i.e. an
              estimate, not a fact. The real SQL could differ.

Category rubric (Mongo-oriented; see docs/query_types.md):
  1 direct             $match/$group/$project/$sort/$limit, at most one simple $lookup.
  2 special_transform  deterministic and feasible, but needs a $lookup chain, $unwind, pivot, window/bucket stages,
                       date/string parsing, per-group top-N, set logic, median via $sortArray, etc.
  3 difficult          feasible but underspecified, tie/LIMIT-dependent, external-knowledge doc, multi-step procedural
                       (gaps-and-islands, retention, rolling/running over densified calendars, FIFO allocation,
                       graph walks) or many differently-shaped golds.
  4 not_faithful       cannot be certified as a faithful conversion: gold CSV is not reproducible from the shipped
                       gold SQL at value level, so no reference procedure is known.
Orthogonal flags (not categories): gold_sql_disagrees, ext_knowledge, view_possible, big_table,
odd_field_names, needs_mentor.
"""
import csv
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATS = {1: "direct", 2: "special_transform", 3: "difficult_ambiguous", 4: "not_faithful"}

# token legend for constructs
TOK = {"J": "join", "F": "filter", "P": "projection", "S": "sort", "L": "limit", "A": "aggregation", "G": "group_by",
       "H": "having", "SQ": "subquery", "CTE": "cte", "SET": "set_operation", "DT": "date_time", "W": "window",
       "STR": "string_parsing", "CASE": "case_expression", "REC": "recursion", "MED": "median_percentile",
       "PIV": "pivot", "RUN": "running_or_rolling", "TOPN": "top_n_per_group", "DENS": "calendar_densification",
       "MATH": "non_trivial_math", "ISL": "gaps_and_islands", "ALLOC": "sequential_allocation", "BUCK": "bucketing",
       "SELF": "self_join_or_graph"}

# id: (tables, tokens, category, flags, note)  -- J<n> = approximate number of joins
D = {
 "local002": ("orders,order_items,products,product_category_name_translation", "J3 F A G DT W RUN MATH", 3, "mentor", "Linear-regression forecast + 5-day symmetric MA; regression spec (x variable, 'toy' category) unstated; 3 gold shapes."),
 "local003": ("orders,order_items,customers", "J2 CTE A G W CASE BUCK S", 4, "gold_unreliable ext_knowledge mentor", "RFM.md. Gold CSV != gold SQL result (labels 'Potential Loyalists' vs 'Potentital', missing segments, values). SQL also did not finish in 25+ min on SQLite."),
 "local004": ("customers,orders,order_payments", "J2 CTE A G DT CASE S L", 2, "", "Top-3 customers by avg payment; lifespan weeks from timestamp text."),
 "local007": ("player", "F A DT MATH STR", 3, "mentor", "Career span from debut/final_game text dates with year+month/12+day/365 and per-part rounding; rounding rule ambiguous (3 golds)."),
 "local008": ("batting,player", "J1 CTE A G SET", 2, "", "UNION of per-stat maxima (games, runs, hits, HR) then names; golds select different columns."),
 "local009": ("flights,airports_data", "J2 F A MATH STR S L", 2, "ext_knowledge", "Haversine.md. airports_data.city / coordinates are text (JSON-like / '(x,y)') needing parsing; trig ops exist in MQL."),
 "local010": ("flights,airports_data", "J2 A G MATH BUCK SQ", 3, "ext_knowledge mentor", "Unordered unique city pairs, avg distance, 1000-km bins, then smallest bin; pair definition and ties unclear."),
 "local015": ("collisions,parties,victims", "J2 F A CASE", 3, "mentor", "Helmet use derived from coded text safety-equipment fields; 5 golds suggest definition variants."),
 "local017": ("collisions", "CTE A G H W DT SET", 2, "", "Top-2 causes per year (ROW_NUMBER) compared across years via GROUP_CONCAT string."),
 "local018": ("collisions", "F A G DT SQ", 2, "", "Share of top 2021 cause in 2021 vs 2011 (ratio-of-ratios)."),
 "local019": ("Matches,Wrestlers,Belts,Cards,Events,Locations,Promotions", "J4 F CTE W STR S L", 2, "big_table", "Shortest NXT title match; duration is text, title_change filter, two wrestler names. Matches is 540k rows."),
 "local026": ("ball_by_ball,batsman_scored,extra_runs,match", "J3 A G TOPN S L", 2, "", "Over-level run sums, max over per match, then top 3 bowlers."),
 "local020": ("wicket_taken,ball_by_ball,batsman_scored,extra_runs,player", "J3 A G S L", 3, "mentor", "'Bowling average' (runs per wicket) undefined: which dismissals/extras count."),
 "local021": ("batsman_scored,ball_by_ball", "J1 A G H SQ", 2, "", "Per-match striker totals >50, then average of their career totals."),
 "local022": ("player,player_match,match,batsman_scored,ball_by_ball", "J4 CTE A G H CASE S", 2, "", "Players with >=100 runs in a match for the losing team."),
 "local023": ("player,player_match,match,batsman_scored,ball_by_ball", "J2 CTE A G SQ S L", 2, "", "Season-5 avg runs per match, top 5 with batting average. CONVERTED: shipped gold SQL not value-faithful, but reconstructed reference (per-match runs + dismissals) reproduces local023_a.csv."),
 "local024": ("player,player_match,batsman_scored,ball_by_ball,match", "J3 A G S L", 2, "", "Per-player avg then per-country avg of those (avg of averages)."),
 "local025": ("ball_by_ball,batsman_scored,extra_runs", "J2 A G TOPN SQ", 2, "", "Max over total per match, then average; 5 golds."),
 "local028": ("olist_orders", "F A G DT PIV", 2, "", "Delivered orders per month, one column per year (pivot)."),
 "local031": ("olist_orders", "F A G DT SQ", 2, "", "Lowest-volume year then its max month."),
 "local029": ("olist_orders,olist_customers,olist_order_payments", "J2 CTE A G S L", 2, "gold_sql_disagrees", "Top 3 customers by delivered orders + avg payment, city, state. Shipped SQL differs in shape from gold CSV."),
 "local030": ("olist_orders,olist_customers,olist_order_payments", "J2 F A G S L SQ", 2, "", "5 lowest-payment cities, then averages."),
 "local032": ("olist_order_items,olist_orders,olist_customers,olist_sellers,olist_order_reviews", "J4 F A G SET", 3, "mentor", "Four 'best seller' categories labelled in rows; 12 golds, ties and labeling unspecified."),
 "local034": ("olist_order_payments,olist_order_items,olist_products", "J2 A G TOPN SQ", 2, "", "Top payment type count per category, then average."),
 "local037": ("olist_order_payments,olist_order_items,olist_products", "J2 A G TOPN S L", 2, "", "Per-category top payment type, then top 3 categories."),
 "local035": ("olist_geolocation", "W STR MATH S", 3, "ext_knowledge big_table mentor", "spherical_law.md. 1,000,163 rows ordered by 5 columns, LAG distance, max; ties in ordering; large sort on small tier."),
 "local038": ("actor,film,film_actor,film_category,category,language", "J5 F A G S L", 2, "", "Actor with most English children's films, rating G/PG, length<=120, 2000-2010."),
 "local039": ("address,category,city,customer,film,film_category,inventory,rental", "J7 F A G DT S L", 2, "case_insensitive_like", "Rental hours = return-rental from text timestamps; city LIKE 'A%' or '%-%'. SQLite LIKE is case-insensitive."),
 "local040": ("trees,income_trees", "J1 F A G S L", 3, "mentor big_table", "'Combined dataset ... filling missing ZIP values' is underspecified; 4 golds, 690k-row trees."),
 "local041": ("trees", "F A", 1, "big_table pilot_done", "PILOT: validated, see docs/pilot_conversion.md."),
 "local049": ("companies_dates,companies_industries", "J1 F A G DT S L", 2, "", "Top industry 2019-2021, avg new unicorns per year."),
 "local054": ("customers,invoices,invoice_items,tracks,albums,artists", "J5 A G H SQ", 2, "", "Best-selling artist, customers spending <$1 on their albums."),
 "local055": ("customers,invoices,invoice_items,tracks,albums,artists", "J5 A G SQ", 2, "", "Top and bottom artist (alphabetical tiebreak), per-customer spend, averages."),
 "local198": ("customers,invoices", "J1 A G H MED", 2, "", "Median of per-country totals where customers>4; exact median via $sortArray (MQL $median is approximate)."),
 "local056": ("customer,payment", "J1 A G DT W S L", 2, "", "Avg monthly change in payment per customer (LAG)."),
 "local058": ("hardware_dim_product,hardware_fact_sales_monthly", "J1 CTE A G DT S", 2, "big_table", "Unique products per segment, YoY % change; 971k-row fact table."),
 "local059": ("hardware_dim_product,hardware_fact_sales_monthly", "J1 F A G TOPN SQ", 2, "big_table", "Top-3 products per division, overall average quantity."),
 "local060": ("sales,customers,countries,times,promotions,products", "J5 F A G W TOPN BUCK", 3, "mentor big_table view_possible", "City growth >=20%, top 20% products, share-of-sales difference; 2 golds with different column sets."),
 "local063": ("sales,customers,countries,times,products", "J4 F A G W TOPN", 3, "mentor big_table", "Same family as local060 with promo_id=999; 5 golds."),
 "local061": ("sales,times,countries,customers,currency,promotions,channels", "J5 F A G MATH DT", 3, "ext_knowledge big_table", "projection_calculation.md: 2019/2020 growth rate -> 2021 projection -> USD."),
 "local050": ("sales,times,countries,customers,currency,promotions,channels", "J5 F A G MATH MED", 3, "ext_knowledge big_table", "Same family as local061, median of monthly averages."),
 "local062": ("customers,countries,sales,costs,times", "J4 F A G BUCK", 2, "big_table view_possible", "Ten equal-width profit buckets; 'profits' is a view (quantity*(price-cost))."),
 "local067": ("customers,countries,sales,costs,times", "J4 F A G BUCK W", 3, "mentor big_table view_possible", "'Ten evenly divided tiers' = NTILE-like vs equal-width ambiguity."),
 "local070": ("cities", "F DT ISL STR S", 3, "mentor", "Shortest/longest streaks of consecutive dates; city-name capitalization; per-date single record."),
 "local071": ("cities", "F DT ISL G", 3, "", "Longest consecutive insert-date streak per country."),
 "local072": ("cities,cities_countries", "F A G DT ISL", 3, "", "Country with 9 insert days; longest streak; capital-city share."),
 "local068": ("cities", "F A G DT RUN", 2, "", "Monthly counts, cumulative per month across years, YoY growth."),
 "local073": ("pizza_clean_customer_orders,pizza_names,pizza_recipes,pizza_toppings", "J3 STR SET S A", 3, "mentor", "Comma-separated topping lists: recipe - exclusions + extras, sorted string with '2x' marker rules."),
 "local066": ("pizza_clean_customer_orders,pizza_recipes,pizza_toppings,pizza_clean_runner_orders", "J2 CTE W SET STR A G", 4, "gold_unreliable", "Shipped SQL disagrees with gold CSV on values (not just shape); no reference procedure."),
 "local065": ("pizza_clean_customer_orders,pizza_clean_runner_orders", "J2 F CASE STR A", 2, "", "Recursive CTE only splits extras strings ($split replaces it); price per pizza type plus $1 extras; exclude cancelled."),
 "local074": ("customer_transactions", "A G DT RUN DENS", 3, "", "Month-end balance incl. months with no activity (densify)."),
 "local064": ("customer_transactions", "F A G DT CASE SQ", 2, "", "2020 month-end balances; best/worst month; difference of averages."),
 "local297": ("customer_transactions", "A G DT RUN W CASE", 2, "", "Monthly net, cumulative balance, latest-month growth rate. AMBIGUOUS GOLD (2 accepted answers: 44.87 filter-all vs 53.00 deposit/withdrawal-only); variant B implemented."),
 "local298": ("customer_transactions", "A G DT RUN W", 2, "", "Prior-month total balance clamped at 0; first month baseline only."),
 "local299": ("customer_transactions", "A G DT RUN W DENS", 3, "", "Daily running balance, 30-day rolling average, monthly max, sum."),
 "local300": ("customer_transactions", "A G DT RUN DENS", 3, "", "Daily balances with carry-forward between first and last txn, monthly sums of per-customer max."),
 "local075": ("shopping_cart_events,shopping_cart_page_hierarchy", "J5 F CTE A G CASE", 2, "", "Views / cart adds / abandoned / purchases per product (gold SQL)."),
 "local077": ("interest_metrics,interest_map", "J1 F A G W RUN TOPN", 2, "", "Monthly max avg composition, 3-month rolling average, prior-month columns."),
 "local078": ("interest_metrics,interest_map", "J1 CTE W SET L S", 2, "", "Top/bottom 10 interests by max composition (RANK + UNION)."),
 "local081": ("customers,orders,order_details,customergroupthreshold", "J3 F A G BUCK", 2, "", "1998 spend per customer, range-join to thresholds, percentage per group."),
 "local085": ("orders,employees", "J1 A G H CASE S L", 2, "", "Employees with >50 orders, top 3 late-order %."),
 "local096": ("Movie,M_Cast,Person", "J2 A G STR SQ", 2, "", "Year = last 4 chars of Movie.year; films with no male/unknown actors per year."),
 "local097": ("Movie", "A G W S L", 2, "", "Ten-year window count starting at each distinct year (range window)."),
 "local098": ("Movie,M_Cast", "A G W ISL", 3, "", "Actors with no gap >3 years between film years."),
 "local099": ("M_Cast,M_Director,Person", "J4 CTE A G CASE", 2, "", "Actors with more films with Yash Chopra than any other director."),
 "local100": ("Person,M_Cast", "J2 SELF SQ", 3, "mentor", "'Shahrukh number' 2: two-hop co-actor graph, excluding direct co-actors."),
 "local114": ("web_accounts,web_orders,web_sales_reps,web_region", "J3 A G TOPN", 2, "", "Per-region totals plus all top reps (ties included)."),
 "local128": ("Bowlers,Bowler_Scores,Tourney_Matches,Tournaments", "J3 F A G H SET", 2, "", "Bowlers with winning handicap<=190 at all three venues."),
 "local130": ("Students,Student_Schedules,Classes,Subjects", "J3 F A W", 3, "mentor", "Quintile by counting grades >= each grade / total (custom definition)."),
 "local131": ("Musical_Preferences,Musical_Styles", "J1 A G H CASE PIV S", 2, "gold_sql_disagrees", "Per-style counts of 1st/2nd/3rd; shipped SQL gives 20 rows, gold CSV 25 (zero-preference styles)."),
 "local133": ("Musical_Preferences,Musical_Styles", "J1 A G CASE S", 1, "", "Weighted 3/2/1 score per style."),
 "local132": ("Entertainers,Entertainer_Styles,Customers,Musical_Preferences", "J3 F A G SELF", 3, "mentor", "Pairs matching first/second style in either order; 'up to three' semantics."),
 "local141": ("salesorderheader,SalesPersonQuotaHistory,salesperson", "J2 A G DT", 2, "", "Annual sales vs annual quota."),
 "local152": ("director_mapping,names,movies,ratings", "J3 A G W DT S L", 2, "", "Top 9 directors by movie count + avg inter-movie duration (date LAG)."),
 "local230": ("genre,movies,ratings,director_mapping,names", "J4 F A G TOPN SQ", 2, "", "Top 3 genres (rating>8) then top 4 directors within."),
 "local156": ("bitcoin_transactions,bitcoin_members", "J1 A G DT W", 2, "", "Annual avg price by region, rank, YoY change."),
 "local157": ("bitcoin_prices", "F STR W DT CASE", 3, "", "Volume 'K'/'M'/'-' parsing; previous non-zero volume (locf); text dates."),
 "local163": ("university_faculty", "CTE A G SQ", 2, "", "Faculty closest to rank average salary (min abs diff; ties)."),
 "local168": ("job_postings_fact,skills_job_dim,skills_dim", "J2 F A G TOPN SQ", 2, "mentor", "Top-3 skills for remote Data Analyst jobs, avg salary. PARTIAL: only 1 of 5 golds reproduced (gold d); skills_dim truncated (26 rows) and skills_job_dim has unnamed skills."),
 "local169": ("legislators,legislators_terms", "J1 F A G DT DENS", 3, "", "Retention curve over 20 years, all periods shown."),
 "local171": ("legislators,legislators_terms", "J1 F A G DT DENS", 3, "", "Active on Dec 31 per year-since-first-term (31..49)."),
 "local167": ("legislators,legislators_terms", "J1 F A G DT S L", 3, "", "Terms containing Dec 31; first-represented state."),
 "local170": ("legislators,legislators_terms", "J1 F A G DT DENS H", 3, "", "Retention at 0/2/..10 years, both genders >0."),
 "local193": ("customer,payment", "J1 A G DT", 2, "", "First 7/30 days by exact timestamp differences."),
 "local194": ("film,film_actor,actor,inventory,rental,payment", "J5 A G TOPN", 3, "big_table", "Film revenue split equally across actors; top 3 per actor."),
 "local195": ("actor,film_actor,film,inventory,rental,customer", "J5 A G S L SQ", 2, "", "Customers who rented films with top-5 actors, as %."),
 "local196": ("customer,payment,rental,inventory,film", "J4 A G W", 2, "", "Rating of first rented film (earliest payment) -> averages."),
 "local197": ("payment,customer", "J1 CTE A G W DT L S", 2, "", "Top 10 payers, largest month-over-month change (LAG)."),
 "local199": ("rental,staff", "J1 CTE A G W DT", 2, "", "Peak month per store from staff-created rentals."),
 "local201": ("word_list", "F STR G A S L", 2, "big_table", "Anagram groups via sorted-character signature ($sortArray); 373,804 words, any starting letter for the anagram partners."),
 "local202": ("alien_data", "A G CASE S L SQ", 2, "", "Top-10 states by population; friendly% > hostile%, avg age >200."),
 "local209": ("orders,stores,deliveries", "J2 A G S L", 2, "big_table", "Top store, share of orders DELIVERED."),
 "local210": ("hubs,stores,orders", "J2 CTE A G DT", 2, "gold_sql_disagrees big_table", "Hubs with >20% Feb->Mar finished-order increase; shipped SQL's column shape differs from gold."),
 "local212": ("deliveries,drivers,orders", "J2 A G S L", 2, "big_table", "Top-5 drivers by average daily deliveries."),
 "local218": ("Match,Team", "A G MED", 2, "", "Median of each team's max season goals."),
 "local219": ("League,Match,Team,Player,match_view", "J10 CTE A G W SET CASE", 3, "view_possible", "Fewest-win team per league incl. zero wins; gold SQL is 26 joins and uses a view."),
 "local221": ("Match,Team", "J1 A G CASE SET S L", 2, "", "Top 10 teams by wins across home/away."),
 "local220": ("Match,Player", "J1 A G SET", 3, "mentor", "Players in most winning / losing matches from 22 player columns (X1..11,Y1..11); 9 golds."),
 "local228": ("batsman_scored,wicket_taken,ball_by_ball,match,player", "J4 A G TOPN SET", 3, "", "Top 3 batsmen and top 3 bowlers per season, matched by rank position."),
 "local229": ("batsman_scored,ball_by_ball,player_match", "J3 A G TOPN", 3, "mentor", "'Partnership runs' not defined in schema; pair ordering rules."),
 "local244": ("Track,InvoiceLine", "J1 CASE A G", 2, "ext_knowledge", "music_length_type.md: short/medium/long thresholds, min/max minutes, revenue."),
 "local253": ("SalaryDataset", "STR A G TOPN SQ", 3, "mentor", "Strip non-numeric chars from salary text; top 5 companies in 4 cities vs national average."),
 "local258": ("wicket_taken,ball_by_ball,batsman_scored,extra_runs,match", "J4 A G TOPN STR", 3, "ext_knowledge", "Wickets, economy, strike rate, best-bowling string 'w-r'."),
 "local259": ("player,player_match,ball_by_ball,batsman_scored,wicket_taken,match", "J5 A G CASE TOPN", 3, "ext_knowledge", "17+ per-player metrics."),
 "local262": ("model,model_score,solution,stack_ok", "J3 A G SQ", 3, "view_possible mentor", "'Stack' vs non-'Stack' test-score comparison; likely relies on view stack_ok."),
 "local263": ("model,model_score,stack_ok", "J3 A G TOPN SQ", 3, "view_possible", "strong/soft status per model, most frequent L1_model per status."),
 "local264": ("model", "A G S L", 2, "mentor", "L1_model frequency; 'traditional vs Stack' framing unclear."),
 "local269": ("packaging,packaging_relations", "REC A G", 3, "", "Fully expand nested packaging, leaf quantities (path-product); $graphLookup cannot multiply along paths -> fixed-depth $lookup unrolling or $reduce."),
 "local270": ("packaging,packaging_relations,packaging", "REC J A F", 3, "", "Top-level containers with an item whose nested total >500."),
 "local272": ("orderlines,inventory,locations,orders", "J3 F ALLOC S", 3, "", "FIFO allocation of inventory to order 423 lines (running remaining quantity)."),
 "local273": ("orders,orderlines,inventory,products", "J3 ALLOC A G S", 3, "", "FIFO pick percentage per product."),
 "local274": ("picking_line,orders,orderlines,products", "J3 F ALLOC A G", 3, "", "Products picked for order 421, FIFO average units."),
 "local275": ("monthly_sales,products", "W RUN MATH F", 3, "ext_knowledge view_possible", "calculation_method.md: centered moving average seasonality ratio >2 in all of 2017."),
 "local277": ("monthly_sales,products", "W MATH A F", 3, "ext_knowledge mentor", "Weighted-regression forecast for products 4160/7790; method per doc."),
 "local279": ("inventory,monthly_orders,product_minimums", "J2 REC DT A", 3, "view_possible mentor", "Recursive month-by-month restock model ($reduce state); depends on view monthly_orders."),
 "local283": ("Match,Team,League,Country", "J3 A G CASE TOPN", 2, "", "Season champion by 3/1/0 points."),
 "local284": ("veg_loss_rate_df", "A MATH CASE", 2, "odd_field_names mentor", "Population vs sample stddev not stated; field 'loss_rate_%'."),
 "local285": ("veg_txn_df,veg_cat,veg_whsle_df,veg_loss_rate_df", "J3 A G DT MATH", 2, "odd_field_names big_table", "Yearly per-category wholesale/selling stats; fields like 'qty_sold(kg)', 'whsle_px_rmb-kg'; veg_txn_df 878k rows."),
 "local286": ("order_items,orders,sellers,products,product_category_name_translation,order_reviews", "J5 A G DT TOPN", 3, "mentor", "Seller performance report; 9 golds; 'highlight top category' unspecified."),
 "local301": ("cleaned_weekly_sales", "SET CASE DT A S", 2, "", "Sales 4 weeks before/after June 15 per year (gold SQL)."),
 "local302": ("cleaned_weekly_sales", "A G DT SET", 2, "", "12 weeks before/after June 15 2020 by attribute type."),
 "local329": ("activity_log", "A G W SQ", 2, "", "Sessions with /regist/input before /regist/confirm."),
 "local330": ("activity_log", "A G W", 2, "", "Landing/exit pages per session counted once."),
 "local331": ("activity_log", "A G W S L", 2, "mentor", "Third page after two consecutive '/detail' visits. UNRESOLVED: 16-variant search matched none of the 3 golds; 7 duplicated session streams (rowid offset 32) make ordering ambiguous."),
 "local358": ("mst_users", "CASE DT A G", 3, "mentor", "Age bucket requires a reference date not given; the two golds disagree on the counts."),
 "local360": ("activity_log", "A G W F S", 3, "", "Per-session events before first /detail or /complete with non-empty search_type; min across sessions."),
 "local344": ("races,lap_positions,pit_stops,retirements,results", "J4 W CASE A G", 3, "ext_knowledge", "f1_overtake.md categorisation of position changes."),
 "local336": ("races,lap_positions,pit_stops,retirements,results", "J4 F W CASE A G", 3, "ext_knowledge", "First-five-laps overtakes by category."),
 "local335": ("results,races,constructors,drivers", "J3 F A G SQ", 2, "", "Constructors with most seasons holding the lowest point-scoring driver total."),
 "local309": ("constructors,drivers,races,results", "J3 CTE A G TOPN SET", 4, "gold_unreliable", "Shipped SQL sums points per year; gold CSV differs in 8 of 75 years (looks like official champions) -> no reference procedure."),
 "local310": ("results,races", "J1 A G TOPN S L", 2, "", "Three years minimising (best driver + best constructor points)."),
 "local311": ("results,drivers,constructors,races", "J3 A G TOPN S L", 2, "", "Top 3 constructors by best-driver + team points, with years."),
 "local354": ("results,races,drivers,constructors", "J3 F W A G", 3, "", "1950s drivers with same constructor in first and last race of a season, >=2 rounds; 8 golds."),
 "local355": ("results,races,drivers", "J2 W ISL A", 3, "", "Races missed (gaps in rounds), constructor switch across hiatus; 5 golds."),
 "local356": ("lap_positions,pit_stops,retirements,drivers,races", "J4 W CASE A G", 3, "", "Net on-track overtakes excluding pit, retirement and lap-1; 4 golds."),
}

# ---------------------------------------------------------------------------
manifest = [json.loads(l) for l in open(ROOT / "data/interim/spider2_local_manifest.jsonl", encoding="utf-8")]
ids = [m["instance_id"] for m in manifest]
assert set(ids) == set(D) and len(ids) == len(D) == 135, (set(ids) ^ set(D))

# row counts / view names / odd field names from the SQLite files (read-only)
meta = {}
for db, d in {m["db"]: m for m in manifest}.items():
    c = sqlite3.connect("file:" + (ROOT / d["database_path"]).as_posix() + "?mode=ro", uri=True)
    t = {}
    for name, ty in c.execute("select name,type from sqlite_master where type in ('table','view')").fetchall():
        try:
            cols = [r[1] for r in c.execute(f'pragma table_info("{name}")')]
        except Exception:
            cols = []
        t[name.lower()] = {"type": ty, "cols": cols}
    meta[db] = t

# SQL hints for the 24 gold-SQL ids
def sql_hints(path):
    s = open(path, encoding="utf-8").read()
    u = s.upper()
    return {"joins_in_sql": len(re.findall(r"\bJOIN\b", u)), "selects_in_sql": len(re.findall(r"\bSELECT\b", u)),
            "has_window": bool(re.search(r"\bOVER\s*\(", u)), "has_recursive": "RECURSIVE" in u}

out = []
for m in manifest:
    iid = m["instance_id"]
    tables, toks, cat, flags, note = D[iid]
    flags = set(flags.split())  # manual flags; shapes/views/ext-knowledge are recomputed below
    flags.discard("multi_gold_shapes")
    toks = toks.split()
    tl = [x for x in tables.split(",") if x]
    db_meta = meta[m["db"]]
    # auto flags
    if m["external_knowledge"]:
        flags.add("ext_knowledge")
    if any(db_meta.get(t.lower(), {}).get("type") == "view" for t in tl):
        flags.add("view_dependent")
    shapes = []
    for p in m["gold_result_paths"]:
        df = list(csv.reader(open(ROOT / p, encoding="utf-8", newline="")))
        shapes.append([max(len(df) - 1, 0), len(df[0]) if df else 0])
    if len({tuple(s) for s in shapes}) > 1:
        flags.add("multi_gold_shapes")
    for t in tl:
        for col in db_meta.get(t.lower(), {}).get("cols", []):
            if re.search(r"[^A-Za-z0-9_]", col):
                flags.add("odd_field_names")
    if any(x in flags for x in ("gold_unreliable", "gold_sql_disagrees")) or "mentor" in flags:
        flags.add("needs_mentor")
    flags.discard("mentor")
    row = {
        "instance_id": iid, "db": m["db"], "question": m["question"],
        "basis": "gold_sql" if m["gold_sql_path"] else "inferred_from_question",
        "tables": tl,
        "constructs": [TOK[re.match(r"[A-Z]+", t).group()] for t in toks],
        "approx_joins": next((int(t[1:]) for t in toks if re.fullmatch(r"J\d+", t)), 0),
        "category": cat, "category_name": CATS[cat], "flags": sorted(flags), "notes": note,
        "n_gold_csv": len(m["gold_result_paths"]), "gold_shapes_rows_cols": shapes,
        "gold_eval_config": {"condition_cols": m["gold_eval_config"]["condition_cols"], "ignore_order": m["gold_eval_config"]["ignore_order"]},
        "external_knowledge": m["external_knowledge"],
    }
    if m["gold_sql_path"]:
        row["gold_sql_hints"] = sql_hints(ROOT / m["gold_sql_path"])
    out.append(row)

p = ROOT / "data/interim/spider2_local_query_patterns.jsonl"
p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")

# ---- counts (computed, not hand tallied) ----
from collections import Counter
print("wrote", p, len(out))
print("category:", {f"{k}:{CATS[k]}": v for k, v in sorted(Counter(r["category"] for r in out).items())})
print("basis:", dict(Counter(r["basis"] for r in out)))
print("flags:", dict(Counter(f for r in out for f in r["flags"])))
print("category x basis:", dict(Counter((r["category"], r["basis"]) for r in out)))
print("construct frequency:", dict(Counter(c for r in out for c in set(r["constructs"])).most_common()))
print("by db:", dict(Counter(r["db"] for r in out)))
print("n_gold_csv>1:", sum(r["n_gold_csv"] > 1 for r in out), "| single table:", sum(len(r["tables"]) == 1 for r in out))
print("needs_mentor ids:", [r["instance_id"] for r in out if "needs_mentor" in r["flags"]])
for k in (1, 4):
    print(f"cat{k} ids:", [r["instance_id"] for r in out if r["category"] == k])
