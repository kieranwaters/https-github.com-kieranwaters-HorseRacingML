# HorseRacingML

ASP.NET Core (net8.0) app that scrapes horse racing results and live Betfair markets,
stores them in SQL Server, and trains/predicts with ML.NET and TensorFlow.

## Database: where it lives now

The database is a normal local SQL Server database. Nothing about the app requires a
remote or permanent server, and no `.mdf`/`.bak` file was ever checked into the repo.

Two things were missing after cloning:

1. a SQL Server instance on the machine, and
2. the schema itself, which used to be created by hand outside source control.

Both are now handled inside the repo:

| File | Purpose |
| --- | --- |
| `HorseRacingML/Data/Schema.sql` | Full schema, reconstructed from every query in `Data/` and `MachineLearning/`. Idempotent: only creates missing tables/indexes, never drops data. |
| `HorseRacingML/Data/DatabaseBootstrapper.cs` | Runs on startup: creates the database from the connection string if it is missing, then applies the embedded schema. |
| `HorseRacingML/Program.cs` | Calls the bootstrapper before registering `RacingRepository`. |

So a fresh clone + a SQL Server instance = a working, self-populating database.
Historical rows are not in git; the app scrapes them back in (see below).

### 1. Install the runtime and a SQL Server instance

```powershell
winget install --id Microsoft.DotNet.SDK.8 --exact
```

Then pick one SQL Server flavour:

* **SQL Server Express** (installed on this machine; a background service, also supports
  SQL authentication):

  ```powershell
  winget install --id Microsoft.SQLServer.2022.Express --exact
  ```

  `winget` installs a default `SQLEXPRESS` instance, and `appsettings*.json` already
  points at `Data Source=.\SQLEXPRESS`. In SSMS, connect to `.\SQLEXPRESS` with Windows
  Authentication.
* **SQL Server Express LocalDB** (lightest, no service, database file lives under your
  user profile): install `SqlLocalDB.msi` from
  <https://learn.microsoft.com/sql/database-engine/configure-windows/sql-server-express-localdb>,
  then change `Data Source` in both `appsettings*.json` files to
  `Data Source=(localdb)\MSSQLLocalDB`.
* **Docker** (good for macOS/Linux or for keeping SQL Server out of Windows):

  ```powershell
  docker run -d --name horseracing-sql -p 1433:1433 `
    -e ACCEPT_EULA=Y -e MSSQL_SA_PASSWORD="YourStrong!Passw0rd" `
    -v horseracing-data:/var/opt/mssql `
    mcr.microsoft.com/mssql/server:2022-latest
  ```

  and use a connection string such as:

  ```json
  "HorseRacingDb": "Server=localhost,1433;Database=HorseRacingML;User Id=sa;Password=YourStrong!Passw0rd;TrustServerCertificate=True;"
  ```

### 2. Run

```powershell
dotnet run --project HorseRacingML
```

On first start the app creates the `HorseRacingML` database, applies `Data/Schema.sql`,
and then behaves exactly as before. With Express the database lives in
`C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS\MSSQL\DATA`; with LocalDB the
files end up in `%LOCALAPPDATA%\Microsoft\Microsoft SQL Server Local DB\Instances\MSSQLLocalDB`
(`*.mdf`/`*.ldf` are already ignored by `.gitignore`, so they will not be committed).

If you ever need to re-apply the schema by hand:

```powershell
sqlcmd -S ".\SQLEXPRESS" -d HorseRacingML -i HorseRacingML\Data\Schema.sql
```

Troubleshooting:

* `A certificate chain was issued by an authority that is not trusted` — add
  `Encrypt=False` to the connection string (LocalDB uses a self-signed certificate).
* `CREATE DATABASE permission denied` — create an empty `HorseRacingML` database in
  SSMS/Azure Data Studio and run `Data/Schema.sql` against it once; the app can then
  use it without elevated permissions.
* `The term 'sqllocaldb' is not recognized` — LocalDB is not installed yet; use one of
  the install options above.

### Run from Visual Studio Code

The repo ships with a `.vscode` folder, so no extra setup is needed:

1. Install the recommended extensions when VS Code prompts you (or install the
   **C# Dev Kit** extension).
2. Open the repository folder and press <kbd>F5</kbd>.
3. Pick **Launch HorseRacingML (http)** (default) or **Launch HorseRacingML (https)**.

The launch configuration builds `HorseRacingML.sln` first, runs the app with
`ASPNETCORE_ENVIRONMENT=Development`, and opens `http://localhost:5051` in your browser.
The same settings are available through **Run and Debug** or **Terminal → Run Task**
(`build`, `build-app`, `run`, `test`, `test-all`). The **Backfill run styles** launch
configuration runs the `--backfill-run-styles` command described below.

If `F5` reports **“Configured debug type 'coreclr' is not supported”**, the C# extension
is not installed or is disabled. Install `ms-dotnettools.csharp`, reload the VS Code
window, and press `F5` again. Without the debugger, **Terminal → Run Task → run** starts
the app all the same.

### 3. Provide Betfair credentials

No credentials are stored in `appsettings*.json` any more. Provide them with user
secrets (they live outside the repo, under your user profile):

```powershell
dotnet user-secrets set "Betfair:Username" "you@example.com" --project HorseRacingML
dotnet user-secrets set "Betfair:Password" "your-password" --project HorseRacingML
```

`BetfairNavigationService` reads `Betfair:Username` / `Betfair:Password` and fails with a
clear message if either is missing. For non-development environments use the
`Betfair__Username` and `Betfair__Password` environment variables instead.

### 4. Refill the historical data

The old data lived only in the lost database. To rebuild it, start the app and hit
`/Home/ScrapeRaceResults` (`HomeController.ScrapeRaceResults`), which walks backwards
from today and inserts courses, races, runners and results. That action needs Chrome and
network access because it uses Selenium.

### Using a different database engine

SQL Server (LocalDB, Express, Developer, Docker, Azure SQL) can be swapped freely by
changing the `HorseRacingDb` connection string; the schema script is standard T-SQL.
SQLite/PostgreSQL are **not** drop-in replacements: the repository and trainer use
T-SQL specifics such as `IDENTITY`/`SCOPE_IDENTITY()`, `DATE`/`TIME(0)`, `sys.indexes`,
`INFORMATION_SCHEMA` plus `SqlBulkCopy`, so switching engines means rewriting
`RacingRepository` and the dataset queries in `HyperparameterTrainer`.

## Notes

* The old Betfair username/password are still present in the git history of this
  repository. Removing them from the working tree does not remove them from past
  commits, so rotate the Betfair password and, if the repo is public, consider purging
  the history (`git filter-repo`/BFG) and force-pushing.
* The TensorFlow package is the Windows GPU build (`TensorFlow.Redist-Windows-GPU`).
  TensorFlow 2.10 needs a compatible CUDA/cuDNN install.
* The old ML.NET/LightGBM winner-prediction path was removed: it was unused by the
  live betting path and used post-race/SP fields as features.

## Model and validation changes

The training and betting path was corrected to remove result leakage and to make the
live pipeline match the backtest:

* Post-race fields (beaten distance, winning time/speed) are excluded from the model
  features. Live rows take trainer, age, weight and official rating from the horse's
  most recent result; the exact race-card weight/rating is not collected pre-race.
* `ProcessRace` builds features from the state as it existed before the race, then
  updates the rolling state afterwards. A regression test asserts that a race result
  cannot appear in its own features.
* `TrainAI` uses a chronological train/dev/test split. Dev is used for early stopping,
  temperature calibration and model selection; test is reported once and never used for
  selection. The persisted model is trained on train+dev with the dev-selected epoch
  count and the dev temperature. Models that do not beat the dev market baseline are
  recorded as `Rejected` and are not eligible for selection.
* Training and evaluation use every race from `Training:MinimumRaceDate` (default
  `1988-01-01`, the start of the stored history). `Training:RequirePreOffPrice` is
  `false` by default, so races from before the market-data era are included even though
  they have no opening prices. Every runner in those races is kept: an unpriced runner
  has `MarketProbMissing = 1`, zero market features and a zero market offset, so its
  form/pace/draw information still contributes and the race-softmax group matches the
  real field. Set `Training:KeepUnpricedRunners = false` to drop unpriced runners, or
  set `MinimumRaceDate` back to `2017-01-01` with `RequirePreOffPrice` set to `true` to
  restrict training to the priced market era. The full history is also replayed to
  build the rolling feature state, so `minimumRaceDate` must stay `null` when loading
  the dataset.
* `Custom` uses walk-forward folds by default. Set **Validate only from** on the
  page to run a single split instead: each row is trained on every race before that
  date and validated once on races from that date onward (for example `2024-11-21`
  for the last two years), without running the earlier folds. `TestAI` uses the same field-normalisation,
  near-uniform abstention, commission and one-bet-per-race rule as live betting.
* Old weight files containing leaky or non-normalised category features are rejected;
  retrain instead of reusing them.
* Live betting skips races within five minutes of the off, US races, markets with a
  placed-bet record, partial/degenerate predictions and near-uniform probability
  vectors. `PlacedBet` provides idempotency and an audit record.
* Relevant configuration:

  ```json
  "Betting": {
    "MaxKellyFraction": 0.10,
    "CommissionRate": 0.05
  },
  "Training": {
    "ValidationFraction": 0.10,
    "DevFraction": 0.10,
    "TestFraction": 0.10,
    "MinimumRaceDate": "1988-01-01",
    "RequirePreOffPrice": false,
    "KeepUnpricedRunners": true,
    "Objective": "RaceSoftmax",
    "AuxiliaryWinLossWeight": 0.2,
    "AuxiliaryFinishLossWeight": 0.2,
    "UseMarketOffset": true,
    "RaceBatchSize": 64,
    "Seed": 42,
    "SeedCount": 1,
    "RequireMarketEdge": true,
    "EnablePaceFeatures": true
  }
  ```

### Speeding up training

The race-softmax trainer is CPU-bound and runs one `sess.run` per race batch, so the
cheapest speed-ups are:

* Use the **Launch HorseRacingML (Release)** configuration in VS Code (or
  `dotnet run --configuration Release`). Debug builds make the feature-engineering and
  batch-building code in `HyperparameterTrainer` noticeably slower.
* Raise `Training:RaceBatchSize` (default `64`) to `128` or `256`. The **Batch Size**
  column on the Custom page does not affect race-softmax training; `RaceBatchSize` does.
  Larger batches mean fewer TensorFlow calls per epoch.
* Reduce the number of epochs or narrow the hyperparameter grid. Changing `Units` or
  `Layers` has little effect on CPU time; epochs control most of it.
* Use an NVIDIA GPU if available. The project references the TensorFlow 2.10 Windows GPU
  build, which needs a compatible CUDA/cuDNN installation.
* Leave `Training:MaxParallelTrials` at `1`. TensorFlow.NET can report graph/session
  errors when models run in parallel, and four CPU cores are already shared.

## Weather and Going

Weather is now ingested from the free Open-Meteo APIs and added to the race
features as a prior for the surface state. It does not replace the official
`Going`: watering, drainage and clerk-of-the-course decisions remain
unobserved, so the model receives both the official going (when available) and
the weather features.

What is stored:

| Table | Purpose |
| --- | --- |
| `Course.Latitude/Longitude/TimeZoneId` | Location used for the weather request. Applied from the built-in UK/Ireland catalogue. |
| `CourseWeatherDaily` | Raw daily precipitation, rain, temperature, wind, gusts and ET₀, cached by course/day. |
| `RaceWeather` | Per-race rain windows (1/3/7/14 days before the race), temperature/wind/ET₀, days since the last meeting, last going at the course, and a heuristic `WeatherGoingIndex`/`WeatherSoftRisk`. |
| `GoingSnapshot` | Append-only log of the going value seen at each scrape. `UpcomingRaces.Going` is an upsert, so without this table earlier readings are overwritten. |

The historical feature pass joins `RaceWeather`; live predictions fetch and
cache missing days for the course and compute the same features. Race-day rain
is only populated for live rows from the forecast, never for a historical
training row, so the backtest cannot see weather that happened after the
prediction.

Run the backfill once after the historical results are loaded:

```powershell
dotnet run --project HorseRacingML -- --backfill-weather --from 2015-01-01

# Recompute race features from the daily cache only (no API calls)
dotnet run --project HorseRacingML -- --populate-weather-features

# Force refresh: re-download every variable for the selected range, even
# course-years that already have race rows. Use after adding new raw columns.
dotnet run --project HorseRacingML -- --refresh-weather --from 2024-01-01 --to 2024-12-31
```

The normal backfill is idempotent and resumable: it skips a course-year once every
race in it has a `RaceWeather` row, so it will not pick up newly added raw columns
on its own. Use `--refresh-weather` for that, and `--populate-weather-features` to
recompute derived features (frost days, rain anomaly) from `CourseWeatherDaily`
without spending API quota.

The same jobs are available as **Backfill Weather**, **Refresh weather data** and
**Rebuild features from cache** on the Weather Status page. They apply course
coordinates first. Unmatched courses (for example tracks outside the UK/Ireland
catalogue) keep null weather features rather than a fabricated value.

**Weather Status** in the navigation bar (or
`dotnet run --project HorseRacingML -- --weather-status`) reports the two table
sizes, coverage by year, and any courses that still have races but no weather
rows. Use it after each backfill run to see what remains.

Configuration:

```json
"Weather": {
  "Enabled": true,
  "ArchiveBaseUrl": "https://archive-api.open-meteo.com/v1/archive",
  "ForecastBaseUrl": "https://api.open-meteo.com/v1/forecast",
  "RequestTimeoutSeconds": 20,
  "PastDays": 14,
  "ForecastDays": 16,
  "LookbackDays": 20,
  "BackfillFrom": "2010-01-01"
}
```

The free Open-Meteo tier is for non-commercial use (10,000 calls/day) and the
data is CC BY 4.0. The application reports the attribution
`Weather data by Open-Meteo.com (CC BY 4.0)`. For commercial use, configure a
customer endpoint/API key or self-host Open-Meteo.

A year-long request is counted as roughly 26 API calls by Open-Meteo, so a
multi-decade, all-course backfill can exceed the free daily quota. The service
stops cleanly when it receives HTTP 429 and is resumable: rerun the same command
(or the button) on a later day, or narrow `--from`/`--to` to one season at a
time. Coverage improves with each run; races without a weather row keep null
features and a missing indicator.

The ordered going target used by the weather prior is Firm, Good to Firm, Good,
Good to Soft/Yielding, Soft, Heavy. All-weather and dirt surfaces deliberately
receive no turf going estimate; the precipitation features remain available.
Feature engineering also derives `GoingIndex` (the ordinal official going) and
`GoingWeatherDelta` (official going minus the weather-implied going), so a
track that is softer or firmer than recent weather explains — for example
because it was watered — is visible to the model rather than hidden inside the
categorical `Going` value.

Adding the weather keys changes the feature schema, so retrain before expecting
a saved model to use them. Older artifacts still load with their own stored
schema and simply ignore the new columns; a model trained after the backfill
picks them up automatically.

## Race-softmax, dev/test split and new features

The default training objective is now a race-level conditional logit (softmax). Each
race is one softmax group, the market's no-vig log-odds are used as an additive offset,
and the network learns the residual edge over the market. `Training:Objective` can be
set to `Binary` to use the old pointwise sigmoid objective for comparison or for old
models.

### Auxiliary win and finish losses

The shared trunk is also trained with two auxiliary heads. They are training-only
regularisers: the persisted artifact still contains a single output head, so live
inference and the model-file contract are unchanged.

| Setting | Default | Effect |
| --- | --- | --- |
| `Training:AuxiliaryWinLossWeight` | `0.2` | Pointwise sigmoid cross-entropy on the winner label. This complements the listwise race softmax with a per-runner signal and is used by `RaceSoftmax`. Set to `0` to disable. |
| `Training:AuxiliaryFinishLossWeight` | `0.2` | Soft-label sigmoid cross-entropy against the normalised finishing position. The target is `(fieldSize - finishPos) / (fieldSize - 1)`, so the winner is `1` and the last finisher is `0`. Rows without a valid `FinishPos` are masked. Used by both objectives. Set to `0` to disable. |

Both weights are recorded in new model artifacts for provenance. They only affect
training, so changing them requires a retrain but not a new feature schema. The finish
target is derived from the post-race `FinishPos` column and is never added to the
feature schema. A good starting point is `0.1` to `0.5`; run the existing benchmark
commands with different values to check whether the auxiliary tasks improve dev/test
log loss.

`TrainAI` now uses a chronological train/dev/test split rather than a single holdout:

* train: earliest `1 - dev - test` fraction of priced races
* dev: next fraction, used for early stopping, temperature and model selection
* test: most recent fraction, touched only once for the final report

The final model is trained on train+dev using the epoch count selected on dev, with the
dev temperature frozen. It is not retrained on test. `TestAI` trains only on races
*before* the chosen evaluation window, so historical backtests no longer see future
form.

New `MLParameters` columns store the objective, seed, chosen epoch, market-offset scale,
normalization mode, date ranges/race counts, dev and test metrics, and the corresponding
market baselines. `GetBestMLParameter` selects on the dev Brier (falling back to the old
`ValidationBrier` for legacy rows) and ignores rows marked `Rejected`.

Feature engineering now includes:

* a base-rate smoothing prior (`WinRateBeta = 9`, so 0 starts maps to ~0.11 rather than
  0.5) and sample-size (`*Starts`) features for condition-specific stats
* neutral (`0.5`) values plus explicit presence indicators for missing history
* within-race rank/percentile features (`*RankRelative`), market rank/favourite/gap
  features, and `RunnerCountInv`
* market context features (`MarketLogit`, `MarketFavouriteProb`, `MarketProbEntropy`,
  `MarketHerfindahl`, `MarketProbStdDev`, `MarketTop2Mass`, `MarketProbGapTop2`,
  `MarketOverround`) and market-vs-form disagreement features
  (`MarketProbMinusLifetimeWinRate`, `MarketLogitMinusFormLogit`,
  `MarketRankMinusWinRateRank`, `MarketRankMinusAvgNormRank`,
  `MarketRankMinusClassRank`)
* priced-subset evaluation: predictions are renormalised over runners with a pre-off
  price, and races whose winner was unpriced are excluded. This is a diagnostic
  subset, not the live-equivalent metric, because live rows have prices for every
  runner

Validate the new model on the same probability space that live betting uses: predictions
are normalised within each race before temperature fitting, Brier/log loss, and early
stopping. New models also carry `Objective`, `MarketOffsetScale` and `Temperature` in
`aiweights.json`; old weight files continue to load as `Binary` models.

## Run-style / pace features

`RunnerResult.Comment` already contains the historical race comments. They are parsed
into a narrow `RunnerRunStyle` table so the chronological feature pass never has to read
the wide `NVARCHAR(MAX)` comment column:

* `RunStyleParser` classifies the early-position clause into `Led`, `Pressed`, `Chased`,
  `Midfield`, `Rear` or `Unknown`, with separate `HeldUp`, `Keen` and `SlowStart` flags.
  Outcome phrases such as "weakened", "headway" or "no extra" are deliberately ignored.
* `RunnerRunStyle` is versioned by `RunStyleParser.Version`; increase the version after
  changing the rules to make the backfill reclassify old rows.
* The model now receives `FrontRunRateLast1/3/5/10`, `ProminentRateLast*`,
  `HeldUpRateLast*`, `AvgPaceScoreLast*`, `PaceStartsLast*`, `LastRunStyle`, and race
  aggregates such as `FrontRunnerCount`, `LoneSpeedFlag`, `PacePressure`,
  `FieldPaceContention`, `FieldMax/Min/AvgFrontRunRate`, plus field max/min/mean for
  speed, win rate and average finishing position.
* Run style is post-race information. It is added to the horse's history only after the
  race, so a race's own comment can never become a feature of that race.

Backfill historical comments (safe to stop and restart; already-classified rows are
skipped unless the parser version is increased):

```powershell
dotnet run --project HorseRacingML -- --backfill-run-styles
```

For a smoke test, limit the number of rows:

```powershell
dotnet run --project HorseRacingML -- --backfill-run-styles --max-rows 10000
```

The web endpoint `/Home/BackfillRunStyles?maxRows=10000` does the same thing in the
background and reports progress through the existing scraping status message. New
results are classified automatically by `BulkInsertRunnerResults`, `InsertRunnerResult`
and the Kaggle importer. Retrain after enabling these features; models trained before
the feature-schema change are not compatible.

`Training:EnablePaceFeatures` (default `true`) is a controlled ablation switch for the
run-style/field-context block. Benchmark the best saved parameter set twice with the
same split, seed and epochs:

```powershell
# Old feature set
dotnet run --project HorseRacingML -- --benchmark-best --disable-pace-features --output "$env:TEMP\pace-off.json"

# New feature set
dotnet run --project HorseRacingML -- --benchmark-best --output "$env:TEMP\pace-on.json"
```

The benchmark never persists weights. It reports dev and test model metrics, fair-market
and blend metrics, and the deltas against the saved reference parameter row.

## Feature blocks

`Training:FeatureBlocks` selects the feature schema. The default schema is:

```json
"Training": {
  "FeatureBlocks": "Pace,MarketMovement,PriceData,RaceContext,TrackGeometry,Sectionals,Breeding"
}
```

Set a comma-separated list to add or remove blocks. `None` disables every block and
`all` enables every available block. Omitting the setting entirely uses the same default
schema in code (`FeatureBlocks.Default`).

The table lists every block; the default set is the JSON above.

| Block | What it adds |
| --- | --- |
| `Pace` | Run-style / field-pace block. |
| `TrainerForm` | `Trainer*` course/going/distance/surface/class/jockey-combination stats, plus `Age`, `AgeRelative` and `AgeProgression`. Live rows take the trainer and age from the horse's most recent result; UK age increments on 1 January, so the inferred age is exact. |
| `MarketMovement` | Opening-to-shortest drift plus the full observed range. Historical rows use the complete `betting.historical_odds` series (opening, touched low and touched high); live rows use the first, lowest and highest `RunnerFlow` snapshots. |
| `WeatherBalance` | Rain minus evapotranspiration over 3/7 days, a rain/ET₀ ratio, recent rain acceleration and the share of the weekly rain that fell in the last day. Derived from the existing `RaceWeather` columns, so no extra API calls or backfill are needed. |
| `RaceContext` | Course and course×distance categoricals, going-specific draw bias, and the day/race index within a meeting. |
| `FormContext` | Non-finisher rates and a completed-runs average position, layoff versus the horse's own typical gap, second career start, jockey/trainer changes, same going/course/distance as last time, class drop/rise versus last run and the best class won. |
| `PriceData` | Official rating, weight, purse, headgear and medication. Ratings also provide the field-strength aggregates (mean/max/min/range/top-3 average/std-dev, plus rating rank and gap to the field's best) that stand in for a separate class block. The historical columns are populated by the resumable Sporting Life price-data backfill; live rows fall back to the horse's most recent result while the exact race-card values are not collected. |
| `WeatherExtended` | Soil moisture (0–7 cm and 0–28 cm), wind direction, humidity/dew point, frost days, rain hours and rainfall/soil-moisture anomalies versus ten-year calendar normals. Needs the extended Open-Meteo backfill. |
| `TrackGeometry` | Left/right handed, straight course, run-in, festival race and festival meeting. |
| `DeepForm` | Second-up after a 60+ day layoff, peak/last-run speed ratios, bounce flag and head-to-head record against today's field. |
| `MarketLiquidity` | Pre-off Betfair traded volume, market matched amount, runner share of the market, exchange price range, projected SP and available-to-back/lay volume. Imported from Betfair Historical Data; the free BASIC package supplies BSP and one-minute last-traded prices, ADVANCED adds traded volume and best-price depth, and PRO adds projected SP. Missing fields are explicit via `Betfair*Missing` flags. |
| `Sectionals` | Prior-run furlong splits from Racing TV / RaceiQ: finishing-speed percentage, last 1f/2f/3f times, last-3f percentile, early and closing position, pace style, plus field aggregates. The current race's own splits are used only to update history, never as its own features. |
| `Breeding` | Rolling sire, dam and damsire progeny statistics (win rate, average finish, surface/going/distance splits) plus field-relative sire/dam strength. The pedigree is read from the Racing TV public racecard; raw names are not model features. |

The feature columns are always built on the live path, even if a block is disabled in
the current configuration; the block switch only controls which columns enter the trained
schema. That means a model trained with a block can still be scored after a restart with
a different `Training:FeatureBlocks` setting. Any block change changes the feature schema
hash, so retrain before activating a model.

Benchmark every block against the same split and the best saved hyperparameter row with:

```powershell
dotnet run --project HorseRacingML -c Release -- `
  --benchmark-blocks `
  --minimum-race-date 2017-01-01 `
  --require-pre-off-price true `
  --keep-unpriced-runners false `
  --output "$env:TEMP\feature-blocks.json"
```

The history is loaded once with every block enabled; each row then trains on the same
train/dev/test split with only that block's columns added to the current baseline. The
report contains model, closing-market and blend metrics for dev and test, plus the delta
against the baseline row. Use `--blocks TrainerForm,FormContext` to run a subset.
When benchmarking `Sectionals`, use `--minimum-race-date 2023-03-01` so the rows before
the RaceiQ archive do not dominate the comparison with neutral values.

`MarketMovement` and `PriceData` are part of the baseline because each improved both dev
and test when added. `RaceContext` and `TrackGeometry` are also enabled so their
higher-cardinality features can be evaluated with a dedicated hyperparameter sweep.
`Sectionals` and `Breeding` are now part of the default schema as well; the remaining
blocks are still opt-in. The code fallback (`FeatureBlocks.Default`) and the
`appsettings*.json` files use the same list.

### Sectionals and breeding sources

The two previously blocked blocks now have a source:

| Block | Source | Coverage | Notes |
| --- | --- | --- | --- |
| `Sectionals` | [Racing TV results / RaceiQ](https://www.racingtv.com/results) | British races from March 2023; Irish races from January 2024 | Furlong-by-furlong times, finishing-speed percentage, in-running positions and RaceiQ pace percentiles. Free to view on racingtv.com. |
| `Breeding` | [Racing TV racecards and horse profiles](https://www.racingtv.com/racing) | Racecard archive from 2010 onward | Sire, dam and damsire for every declared runner. One racecard request covers the whole field. |

Both backfills use the public JSON endpoints behind racingtv.com. They are serialised
and rate-limited (`RacingTv:RequestDelayMs`, default 300 ms) and are resumable by date:

```powershell
# Furlong splits: March 2023 onward (GB), January 2024 onward (IRE).
dotnet run --project HorseRacingML -c Release -- --backfill-sectionals `
  --from 2023-03-01 --to 2026-12-31

# Sire/dam/damsire for historical runners.
dotnet run --project HorseRacingML -c Release -- --backfill-breeding `
  --from 2010-01-01 --to 2026-12-31
```

Use `--max-days 10` for a smoke test and `--force` to re-fetch races that already have
rows. The same jobs are available as **Backfill Sectionals** and **Backfill Breeding**
in the navigation bar. Check progress with:

```powershell
dotnet run --project HorseRacingML -- --racingtv-status
```

The sectionals archive is much shorter than the result history. Train and benchmark
`Sectionals` from `2023-03-01`; earlier rows receive neutral values and an explicit
`SectionalHasHistory = 0` flag. `Breeding` can start from 2010, when the racecard archive
begins. Both blocks are enabled in the default configuration; run the backfills before
expecting them to add signal. On the live path, a runner with no stored pedigree
is looked up once through the Racing TV horse search/profile endpoints and the result is
cached in `HorsePedigree`; a failed lookup leaves the Breeding features neutral.

This is an unofficial, free public web API rather than a licensed bulk feed. Keep the
request rate conservative and check Racing TV's terms before commercial use. Paid
alternatives with explicit licences include Timeform sectional archives, Total
Performance Data (TPD) for tracking data and the official Weatherbys/Equineline
pedigree services.

### Price-data backfill

Sporting Life result pages expose official rating, prize money, headgear and medication.
The backfill updates existing `RunnerResult`/`Race` rows in place, so it is safe to rerun:

```powershell
dotnet run --project HorseRacingML -c Release -- --backfill-price-data `
  --from 2016-11-14 --to 2026-11-18 --parallelism 6
```

`Medication` is usually empty in the source; `OfficialRating`, `Purse` and `Headgear`
coverage improves as the backfill progresses. The same pass also fills
`OpeningFraction`, `TouchedHighFraction` and `TouchedLowFraction` from the complete
`betting.historical_odds` array on each result page.

To refresh only the market-movement columns (for example after changing the parser),
use the dedicated backfill:

```powershell
dotnet run --project HorseRacingML -c Release -- --backfill-market-movement `
  --from 2016-11-14 --to 2026-11-18 --parallelism 6
```

Both backfills are resumable and update existing rows in place; scrape the results for a
date first if it has no `RunnerResult` rows yet. The source's
`op`/`tchd` comment only carries one touched price; the JSON `historical_odds` array
carries the sequence of observed prices, which is what makes the high/range features
possible. Sporting Life's current results archive starts on **2016-11-14**; earlier
result rows have no full odds series. Rows with only one observed price keep
`TouchedHighFraction` null and the `MarketMoveHighMissing` flag makes that explicit.

### Betfair historical import

`MarketLiquidity` needs Betfair Exchange data. Download the markets you need from the
[Betfair Historical Data site](https://historicdata.betfair.com/).
`--import-betfair-historical` accepts a single file or a directory and reads the
newline-delimited JSON stream format directly, including `.tar.bz2`, `.bz2`, `.tar.gz`
and the individually extracted `.json` files:

```powershell
# Inspect matches without writing to the database.
dotnet run --project HorseRacingML -c Release -- `
  --import-betfair-historical "C:\betfair\2024-01" --dry-run

# Import for real.
dotnet run --project HorseRacingML -c Release -- `
  --import-betfair-historical "C:\betfair\2024-01"
```

The importer supports `.json`, `.jsonl`, `.gz`, `.tar`, `.tar.gz`, `.bz2` and
`.tar.bz2`. Markets are matched to races by date, venue and scheduled off time (20-minute tolerance), and
runners by normalised horse name, so re-importing the same market is idempotent.
Import the Sporting Life results for the same dates first; Betfair markets with no
matching race are skipped. Only WIN markets are written, so an event download that
also contains PLACE markets cannot overwrite the win-market BSP/volume.

What the packages provide:

| Betfair package | Coverage | Fields used |
| --- | --- | --- |
| BASIC (free) | From April 2015 | Market definition, BSP, one-minute last traded price high/low/range |
| ADVANCED (paid) | From April 2015 | BASIC plus traded volume and best available-to-back/lay volume |
| PRO (paid) | From April 2015 | ADVANCED plus the full price ladder and projected SP (`spn`/`spf`) |

The importer writes `Race.BetfairMarketId`, `BetfairMarketName` and `BetfairTotalMatched`,
plus `RunnerResult.BetfairSelectionId`, `BetfairSP`, `BetfairPreOffHigh`, `BetfairPreOffLow`,
`BetfairLastTradedPrice`, `BetfairTradedVolume`, `BetfairProjectedSP`,
`BetfairAvailableToBack` and `BetfairAvailableToLay`.

Betfair historical data starts in **April 2015**, so the exchange range/BSP/volume can be
filled for 2015-04 onward even where Sporting Life's bookmaker odds series is unavailable.
It does not provide the bookmaker `OpeningFraction`/`TouchedHighFraction` columns.

BSP is stored in `RunnerResult.BetfairSP` for evaluation and CLV work, but it is
deliberately **not** a model feature: BSP is only known at the off, so using it in a
pre-off prediction would leak the closing line. The pre-off liquidity and exchange
range columns are safe to use and are exposed through the `MarketLiquidity` block:

```json
"Training": {
  "FeatureBlocks": "Pace,MarketMovement,PriceData,RaceContext,TrackGeometry,Sectionals,Breeding,MarketLiquidity"
}
```

The historical import populates those columns for training and backtesting. There is no
live Betfair API snapshot path yet, so `IsLiveAvailableFeature` rejects the
`Betfair*` columns: a model trained with `MarketLiquidity` is a research/backtest model
and cannot be activated for live scoring. Add a live `listMarketBook` capture that
fills the same columns before enabling the block live.

### Weather-extended backfill

```powershell
dotnet run --project HorseRacingML -c Release -- --backfill-weather `
  --from 2015-01-01 --to 2026-12-31
```

The free Open-Meteo tier is quota-limited; a 429 stops the run cleanly and the same
command resumes later. Weather normals are computed from the stored daily history, so
they improve as the backfill fills in past years.

### Closing-line value

Once `RunnerFlow` contains pre-off snapshots and bets have been placed:

```powershell
dotnet run --project HorseRacingML -- --clv-report
```

Positive CLV means the bet price was better than the last recorded pre-off price. The
same data is needed before live `MarketMovement` drift features can be evaluated.

### Market snapshot scheduler

`Betting:EnableSnapshotScheduler` (default `false`) starts a background timer that runs
the existing market scan every `SnapshotIntervalMinutes` between `SnapshotStartHour` and
`SnapshotEndHour`. The scan records `RunnerFlow` snapshots but does not place bets. Enable
it only when Betfair credentials and Chrome are configured.
