# Quick and Dirty Metric to Imperial Conversions; How to Entertain Yourself as an American Driving in a Metric Country

I was driving in Canada recently, the fun part being all my car's gauges are in imperial (i.e., Freedom 🦅) units (miles per hour, miles, etc..), but road signage is in metric units (Kmph,Km).

So I wanted to figure out how fast in miles per hour I should go in a given spot. The gauges are digital, so they can be made to read metric, but that's too easy. Plus my car doesn't let you switch units while driving
and I kept forgetting to switch the units when I was stopped; so we're stuck estimating.

I remembered from my running days `3 mi ~= 5 Km`, so `10 Km/h ~= 6 mph`. Which means for a known Km/h, Mi/h is given by
```
SPEED_MPH = (6*SPEED_KMPH/10)
```

If the speed limits in 100Km/h, then you should go `60 mi/h`.

Using Google Map's speed estimate, the estimated speed looked close. Note the estimate is often different from the car's speedometer
by 2-3 MPH so it serves as more of a sanity check than anything.

I then remembered the actual distance for`5Km` was closer to `3.1 mi`. That means the quick and dirty estimate above works well at lower speeds, but the error accumulates at higher speeds like on a highway.
This error isn't necessarily harmful since the estimate is lower, but could annoy other drivers.
A more accurate estimation method accounts for the tenths position:
```
SPEED_MPH = (6*SPEED_KMPH/10) + SPEED_KMPH//50
```
This gives a speed of `62 mph`.
This estimate is really close since `100 Km/h ~= 62.1371mph`. The trailing error won't be noticeable until we near 1000+Km/h.
Probably not an issue unless your driving a supersonic jet and need to convert units on the fly.

In real life you should probably just pull over and change your gauges, but this kept me entertained for a little bit while driving.