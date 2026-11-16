City- and event-level results

Identifiers used in all files: city_id is the provider's city code; city_name_zh and city_name_en are the Chinese and English names of the prefecture-level unit; province_en is its province. lat and lon (degrees N and E, WGS84) give a point inside the unit's polygon. Shares are on a 0-1 scale unless the column name ends in _pp (percentage points) or _km (kilometres).

event_catalogue_and_estimates.csv (643 rows, one per analysed city event in 2017-2019)
  city_event_id: event identifier. city_id, city_name_zh, city_name_en, province_en: the city.
  rainfall_system_id: rainfall system (events in adjoining prefectures that overlap in time or are at most one day apart).
  event_start, event_end (YYYY-MM-DD), duration_days: the run of extreme-rain days.
  peak_rain_mm, cumulative_rain_mm: largest daily and total ERA5-Land rainfall of the event (mm). return_level_3yr_mm: the city's 3-year return level of daily rainfall (mm). peak_relative_to_return_level: peak_rain_mm / return_level_3yr_mm.
  change_local_pp, change_inbound_pp, change_outbound_pp, change_at_merchants_pp: mean change in spending over event days relative to the clean pre-event baseline, in percentage points of counterfactual spending (local, inbound, outbound, and local plus inbound at the city's merchants).
  shared_rain_share_S: ex-ante shared-rain share of the city's external demand (prior-year ties, 1981-2016 rainfall), 0-1; defined for the 440 events of 2018-2019.
  partner_demand_hit: share of the city's prior-year external demand from cities in the same rainfall system (0-1).
  effective_origins: effective number of origin cities of external demand (1 / sum of squared weights).
  mean_partner_distance_km: demand-weighted mean distance to origin cities (km).
  in_estimation_sample: 1 for the 364 events in rainfall systems with at least two such events.

city_year_shared_rain_share.csv (672 rows, one per city and network year 2017-2019; 224 cities)
  city_id, city_name_zh, city_name_en, province_en, lat, lon: the city.
  network_year: year of the intercity ties (events of year t use the network of year t - 1).
  n_partners: listed origin cities of the city's inbound spending in the network year.
  shared_rain_share_S, S_from_1981_1998, S_from_1999_2016: shared-rain share (0-1) from the full 1981-2016 rainfall record and from each half.
  effective_origins: as above. mean_partner_distance_km: demand-weighted mean distance to origin cities (km). share_of_external_demand_within_300km: share of external demand from origin cities within 300 km (0-1).

decoupling_distance_by_unit.csv (367 prefecture-level rainfall units with a same-day chance estimate)
  city_name_zh, city_name_en, province_en, lat, lon: the unit.
  decoupling_distance_km, ci_low_km, ci_high_km: distance beyond which other units have extreme rain on fewer than 5% of the unit's extreme-rain days (ERA5-Land, 1981-2016), with its 95% interval from 999 bootstrap draws of the other units (km). Empty for the two units described in note.
  decoupling_distance_cpc_km, cpc_ci_low_km, cpc_ci_high_km: the same with CPC gauge analyses (km); 0 means no CPC decoupling distance (6 units).
  sample_city: 1 for the 224 sample cities with rainfall data.
  note: reason a unit has no ERA5-Land decoupling distance.

east_west_cooperation_pairs.csv (188 rows, one per pair of an eastern and a western unit)
  period: 2021-present (97 pairs in force after the 2021 adjustment of the East-West cooperation programme; used in the article) or 2016-2020 (91 pairs).
  east_province_zh/en, east_unit_zh/en, west_province_zh/en, west_unit_zh/en: the eastern and western province and prefecture-level unit.
  level: administrative level of the pairing (city_to_prefecture, city_to_municipality, municipality_to_prefecture). Flags mark pairs that are not stated as such in one official list: FLAG_derived, list obtained from the 2021 table by removing units the earlier plan did not cover; FLAG_carried_forward, names from the 2010 list whose continuation after 2021 was not separately confirmed; FLAG_duikou, links set up under the 2014 paired-assistance plan for Tibetan areas; province_to_province_FLAG, a province-level pairing without a documented prefecture-level mapping.
  source_url: official or media source of the pairing.

Spending changes are derived from licensed records and are given only as event-level summaries. No daily spending series or city-to-city spending ties are included.
