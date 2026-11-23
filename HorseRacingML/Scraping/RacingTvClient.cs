using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;

namespace HorseRacingML.Scraping
{
    /// <summary>
    /// Thin client for the public endpoints used by racingtv.com. Sectional
    /// times (RaceiQ) are free and backdated to March 2023 for Britain and
    /// January 2024 for Ireland. Racecards also carry sire/dam/damsire for every
    /// declared runner, which is the breeding source.
    ///
    /// The client deliberately serialises requests and leaves a configurable
    /// gap between calls. This is an undocumented public web API, not a
    /// licensed bulk feed: keep the rate low and do not use it to bypass a
    /// paywall.
    /// </summary>
    public sealed class RacingTvClient : IDisposable
    {
        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNameCaseInsensitive = true
        };

        private readonly HttpClient _http;
        private readonly int _requestDelayMs;
        private readonly SemaphoreSlim _requestGate = new(1, 1);

        public RacingTvClient(IConfiguration configuration)
        {
            var baseUrl = configuration.GetValue<string>("RacingTv:BaseUrl");
            if (string.IsNullOrWhiteSpace(baseUrl))
            {
                baseUrl = "https://api.racingtv.com";
            }

            _requestDelayMs = Math.Clamp(
                configuration.GetValue<int?>("RacingTv:RequestDelayMs") ?? 300,
                0,
                10_000);
            int timeoutSeconds = Math.Clamp(
                configuration.GetValue<int?>("RacingTv:TimeoutSeconds") ?? 30,
                5,
                120);

            _http = new HttpClient
            {
                BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/"),
                Timeout = TimeSpan.FromSeconds(timeoutSeconds)
            };
            _http.DefaultRequestHeaders.TryAddWithoutValidation("Accept", "application/json");
            _http.DefaultRequestHeaders.TryAddWithoutValidation("Referer", "https://www.racingtv.com/");
            _http.DefaultRequestHeaders.TryAddWithoutValidation(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36");
            _http.DefaultRequestHeaders.TryAddWithoutValidation("X-Requested-With", "racingtv-web/5.6.0");
        }

        public Task<RacingTvResultsList?> GetResultsAsync(
            DateTime date,
            CancellationToken cancellationToken = default)
            => GetJsonAsync<RacingTvResultsList>(
                $"racing/results/list/{date:yyyy-MM-dd}",
                cancellationToken);

        public Task<RacingTvRacecard?> GetRacecardAsync(
            DateTime date,
            string trackSlug,
            string localTime,
            CancellationToken cancellationToken = default)
            => GetJsonAsync<RacingTvRacecard>(
                $"racing/racecards/{date:yyyy-MM-dd}/{NormalizeSlug(trackSlug)}/{NormalizeTime(localTime)}",
                cancellationToken);

        public Task<RacingTvSectionalRace?> GetSectionalsAsync(
            DateTime date,
            string trackSlug,
            string localTime,
            CancellationToken cancellationToken = default)
            => GetJsonAsync<RacingTvSectionalRace>(
                $"racing/iq-results/{date:yyyy-MM-dd}/{NormalizeSlug(trackSlug)}/{NormalizeTime(localTime)}/sectional",
                cancellationToken);

        public Task<RacingTvHorseProfile?> GetHorseProfileAsync(
            string slug,
            CancellationToken cancellationToken = default)
            => GetJsonAsync<RacingTvHorseProfile>(
                $"racing/horses/{NormalizeSlug(slug)}",
                cancellationToken);

        public Task<RacingTvHorseSearch?> SearchHorsesAsync(
            string name,
            CancellationToken cancellationToken = default)
            => GetJsonAsync<RacingTvHorseSearch>(
                "search/horse?query=" + Uri.EscapeDataString(name ?? string.Empty),
                cancellationToken);

        private async Task<T?> GetJsonAsync<T>(
            string path,
            CancellationToken cancellationToken)
            where T : class
        {
            await _requestGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                for (int attempt = 0; attempt < 3; attempt++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    try
                    {
                        using var response = await _http.GetAsync(path, cancellationToken).ConfigureAwait(false);
                        if (response.StatusCode == HttpStatusCode.NotFound)
                        {
                            return null;
                        }

                        if (response.StatusCode == HttpStatusCode.TooManyRequests)
                        {
                            var wait = ReadRetryAfter(response) ?? TimeSpan.FromSeconds(5 * (attempt + 1));
                            await Task.Delay(wait, cancellationToken).ConfigureAwait(false);
                            continue;
                        }

                        if ((int)response.StatusCode >= 500 && attempt < 2)
                        {
                            await Task.Delay(TimeSpan.FromSeconds(attempt + 1), cancellationToken)
                                .ConfigureAwait(false);
                            continue;
                        }

                        response.EnsureSuccessStatusCode();
                        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken)
                            .ConfigureAwait(false);
                        return await JsonSerializer.DeserializeAsync<T>(stream, JsonOptions, cancellationToken)
                            .ConfigureAwait(false);
                    }
                    catch (HttpRequestException) when (attempt < 2)
                    {
                        await Task.Delay(TimeSpan.FromSeconds(attempt + 1), cancellationToken)
                            .ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (
                        !cancellationToken.IsCancellationRequested && attempt < 2)
                    {
                        // HttpClient timeout, not user cancellation. Retry
                        // instead of aborting a multi-hour backfill.
                        await Task.Delay(TimeSpan.FromSeconds(attempt + 1), cancellationToken)
                            .ConfigureAwait(false);
                    }
                    finally
                    {
                        if (_requestDelayMs > 0)
                        {
                            await Task.Delay(_requestDelayMs, cancellationToken).ConfigureAwait(false);
                        }
                    }
                }

                return null;
            }
            finally
            {
                _requestGate.Release();
            }
        }

        private static TimeSpan? ReadRetryAfter(HttpResponseMessage response)
        {
            var retryAfter = response.Headers.RetryAfter;
            if (retryAfter?.Delta is { } delta)
            {
                return delta;
            }

            if (retryAfter?.Date is { } date)
            {
                var wait = date - DateTimeOffset.UtcNow;
                return wait > TimeSpan.Zero ? wait : TimeSpan.FromSeconds(1);
            }

            return null;
        }

        private static string NormalizeSlug(string? slug)
            => (slug ?? string.Empty).Trim().Trim('/');

        private static string NormalizeTime(string? localTime)
        {
            var text = (localTime ?? string.Empty).Trim().Replace(":", string.Empty);
            return text.Length == 4 ? text : text.PadLeft(4, '0');
        }

        public void Dispose()
        {
            _http.Dispose();
            _requestGate.Dispose();
        }
    }

    public sealed class RacingTvResultsList
    {
        [JsonPropertyName("meetings")]
        public List<RacingTvMeeting> Meetings { get; set; } = new();
    }

    public sealed class RacingTvMeeting
    {
        [JsonPropertyName("track")]
        public RacingTvTrack? Track { get; set; }

        [JsonPropertyName("completed_races")]
        public List<RacingTvRaceSummary> CompletedRaces { get; set; } = new();
    }

    public sealed class RacingTvTrack
    {
        [JsonPropertyName("id")]
        public long? Id { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("slug")]
        public string? Slug { get; set; }

        [JsonPropertyName("country_code")]
        public RacingTvCountryCode? CountryCode { get; set; }
    }

    public sealed class RacingTvCountryCode
    {
        [JsonPropertyName("iso")]
        public string? Iso { get; set; }

        [JsonPropertyName("racing")]
        public string? Racing { get; set; }
    }

    public sealed class RacingTvRaceSummary
    {
        [JsonPropertyName("id")]
        public long? Id { get; set; }

        [JsonPropertyName("title")]
        public string? Title { get; set; }

        [JsonPropertyName("slug")]
        public string? Slug { get; set; }

        [JsonPropertyName("start_time_scheduled")]
        public DateTimeOffset? StartTimeScheduled { get; set; }
    }

    public sealed class RacingTvRacecard
    {
        [JsonPropertyName("race")]
        public RacingTvRaceInfo? Race { get; set; }

        [JsonPropertyName("runners")]
        public List<RacingTvRacecardRunner> Runners { get; set; } = new();
    }

    public sealed class RacingTvRaceInfo
    {
        [JsonPropertyName("id")]
        public long? Id { get; set; }

        [JsonPropertyName("title")]
        public string? Title { get; set; }

        [JsonPropertyName("start_time_scheduled")]
        public DateTimeOffset? StartTimeScheduled { get; set; }
    }

    public sealed class RacingTvRacecardRunner
    {
        [JsonPropertyName("id")]
        public long? Id { get; set; }

        [JsonPropertyName("horse_id")]
        public long? HorseId { get; set; }

        [JsonPropertyName("horse_name")]
        public string? HorseName { get; set; }

        [JsonPropertyName("cloth_number")]
        public int? ClothNumber { get; set; }

        [JsonPropertyName("withdrawn")]
        public bool Withdrawn { get; set; }

        [JsonPropertyName("horse")]
        public RacingTvHorse? Horse { get; set; }
    }

    public sealed class RacingTvHorse
    {
        [JsonPropertyName("id")]
        public long? Id { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("slug")]
        public string? Slug { get; set; }

        [JsonPropertyName("gender")]
        public string? Gender { get; set; }

        [JsonPropertyName("colors")]
        public List<string>? Colors { get; set; }

        [JsonPropertyName("age")]
        public int? Age { get; set; }

        [JsonPropertyName("age_days")]
        public int? AgeDays { get; set; }

        [JsonPropertyName("foaling_date")]
        public DateTime? FoalingDate { get; set; }

        [JsonPropertyName("sire_name")]
        public string? SireName { get; set; }

        [JsonPropertyName("dam_name")]
        public string? DamName { get; set; }

        [JsonPropertyName("dam_sire_name")]
        public string? DamSireName { get; set; }

        [JsonPropertyName("country_code")]
        public RacingTvCountryCode? CountryCode { get; set; }
    }

    public sealed class RacingTvSectionalRace
    {
        [JsonPropertyName("race")]
        public RacingTvSectionalRaceInfo? Race { get; set; }
    }

    public sealed class RacingTvSectionalRaceInfo
    {
        [JsonPropertyName("scheduled_start_time")]
        public DateTimeOffset? ScheduledStartTime { get; set; }

        [JsonPropertyName("time_index_score")]
        public double? TimeIndexScore { get; set; }

        [JsonPropertyName("meeting_average_time_index")]
        public double? MeetingAverageTimeIndex { get; set; }

        [JsonPropertyName("win_time_vs_par")]
        public double? WinTimeVsPar { get; set; }

        [JsonPropertyName("runners")]
        public List<RacingTvSectionalRunner> Runners { get; set; } = new();
    }

    public sealed class RacingTvSectionalRunner
    {
        [JsonPropertyName("id")]
        public long? Id { get; set; }

        [JsonPropertyName("finish_position")]
        public int? FinishPosition { get; set; }

        [JsonPropertyName("finish_text")]
        public string? FinishText { get; set; }

        [JsonPropertyName("cloth_number")]
        public int? ClothNumber { get; set; }

        [JsonPropertyName("barrier_number")]
        public int? BarrierNumber { get; set; }

        [JsonPropertyName("horse_name")]
        public string? HorseName { get; set; }

        [JsonPropertyName("finishing_time")]
        public string? FinishingTime { get; set; }

        [JsonPropertyName("finishing_speed_percentage")]
        public string? FinishingSpeedPercentage { get; set; }

        [JsonPropertyName("sectionals")]
        public List<RacingTvSectionalSplit> Sectionals { get; set; } = new();
    }

    public sealed class RacingTvSectionalSplit
    {
        [JsonPropertyName("sector_number")]
        public int SectorNumber { get; set; }

        [JsonPropertyName("sector_name")]
        public string? SectorName { get; set; }

        [JsonPropertyName("sector_time")]
        public double? SectorTime { get; set; }

        [JsonPropertyName("time_behind_leader")]
        public double? TimeBehindLeader { get; set; }

        [JsonPropertyName("sector_position")]
        public int? SectorPosition { get; set; }

        [JsonPropertyName("sector_pace_category")]
        public string? SectorPaceCategory { get; set; }

        [JsonPropertyName("sector_time_percentile")]
        public double? SectorTimePercentile { get; set; }

        [JsonPropertyName("cumulative_sector_time")]
        public double? CumulativeSectorTime { get; set; }
    }

    public sealed class RacingTvHorseProfile
    {
        [JsonPropertyName("horse")]
        public RacingTvHorse? Horse { get; set; }
    }

    public sealed class RacingTvHorseSearch
    {
        [JsonPropertyName("hits")]
        public List<RacingTvHorseSearchHit> Hits { get; set; } = new();
    }

    public sealed class RacingTvHorseSearchHit
    {
        [JsonPropertyName("type")]
        public string? Type { get; set; }

        [JsonPropertyName("id")]
        public long? Id { get; set; }

        [JsonPropertyName("horse")]
        public RacingTvHorse? Horse { get; set; }
    }
}
