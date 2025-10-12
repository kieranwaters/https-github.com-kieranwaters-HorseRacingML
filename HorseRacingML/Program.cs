using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Scraping;
using HorseRacingML.Services;
using Microsoft.AspNetCore.Hosting;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

var hostingSection = builder.Configuration.GetSection("Hosting");
var httpPort = hostingSection.GetValue("HttpPort", 5051);
var httpsPort = hostingSection.GetValue("HttpsPort", 7251);
var enableHttps = hostingSection.GetValue("EnableHttps", false);

builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenLocalhost(httpPort);

    if (enableHttps)
    {
        options.ListenLocalhost(httpsPort, listenOptions => listenOptions.UseHttps());
    }
});

// Add services to the container.
builder.Services
    .AddControllersWithViews()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
    });

// Register the repository so parsed data can be inserted into the database
builder.Services.AddScoped<RacingRepository>(sp =>
    new RacingRepository(builder.Configuration.GetConnectionString("HorseRacingDb")!));

// TensorFlow trainer for running models on the GPU
builder.Services.AddSingleton<AutomationSettingsService>();
builder.Services.AddSingleton<HyperparameterTrainer>();
builder.Services.AddSingleton<BetfairNavigationService>();
builder.Services.AddTransient<RaceResultsScraper>();
builder.Services.AddSingleton<ScrapingStatusService>();
builder.Services.AddTransient<RaceDataScraper>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

if (enableHttps)
{
    app.UseHttpsRedirection();
}
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();