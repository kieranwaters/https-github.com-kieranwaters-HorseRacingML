using HorseRacingML.Data;
using HorseRacingML.ML;
using HorseRacingML.Scraping;
using HorseRacingML.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();

// Register the repository so parsed data can be inserted into the database
builder.Services.AddScoped<RacingRepository>(sp =>
    new RacingRepository(builder.Configuration.GetConnectionString("HorseRacingDb")!));

// TensorFlow trainer for running models on the GPU
builder.Services.AddSingleton<AutomationSettingsService>();
builder.Services.AddSingleton<HyperparameterTrainer>();
builder.Services.AddSingleton<BetfairNavigationService>();
builder.Services.AddTransient<RaceResultsScraper>();
builder.Services.AddSingleton<ScrapingStatusService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();