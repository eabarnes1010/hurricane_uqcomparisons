# DATA DICTIONARY

## Storm and forcast data
* __ATCF__ : Storm ID, where first 2 letters are the basin ID 
    * AL Atlantic 
    * EP eastern north Pacific
    * CP central north Pacific
* __Name__ : storm name
* __Date__ : Starting date of the forecast [YYYY]
* __Time__ : Starting time of the forecast [hhmmss]
* __ftime(hr)__ : Forecast hour [hr]. The data are included every 12 hr through 168 hours
    * One of {12  24  36  48  60  72  84  96 108 120 132 144 156 168}

## Storm environmental variables
* __DV12__ : previous 12 hr intensity change [kt]
* __SLAT__ : storm latitude from the NHC official forecast [deg]
* __SSTN__ : surface sea temperature [degC]
* __SHDC__ : vertical shear [1/s]
* __DTL__ :  distance to land [km]

## Intensity specific data
* __NCI__ : Number of intensity models included in the consensus (max of 4) [\#]
* __VMXC__ : The max wind of the consensus forecast [kt]
* __VMAX0__: The observed max wind at the time of forecast [kt]
* __OBDV__ : The difference in the observed max wind from the consensus max wind in [kt]. _This is what we are trying to predict for intensity._

#### Difference between the model predicted intensity and the model consensus intensity.
* __DSDV__ : DSHIPS Decay-Statistical Hurricane Intensity Prediction Scheme [kt]
    * Statistical-dynamical model based on standard multiple regression techniques.
    * Climatology, persistence, environmental atmosphere parameters, oceanic input, and an inland decay component.
* __LGDV__ : LGEM Logistic Growth Equation Model [kt]
    * Statistical intensity model based on a simplified dynamical prediction framework.
    * A subset of SHIPS predictors, ocean heat content, and variability of the environment used to determine growth rate maximum wind coefficient.
* __HWDV__ : HWRF Hurricane Weather Research and Forecast system [kt]
    * Nested Grid point (13.5-4.5-1.5km)
* __AVDV__ : GFS Global Forecast System (FV3-GFS) [kt]
    * Finite Volume Cube Sphere (~13km)

## Track specifc data
* __NCT__ : Number of track models in the consensus (max of 4) [\#]
* __LONC__ : Longitude of the consensus forecast [deg]
* __LATC__ : Latitude of the consensus forecast [deg]

#### Difference (east-west) between the model predicted track and the track model consensus.
* __OBDX__ : east-west difference [km] of the observed storm position minus that of the track model consensus. _This is what we are trying to predict for the east-west displacement._
* __AVDX__ : GFS Global Forecast System (FV3-GFS) [km]
    * Finite Volume Cube Sphere (~13km)
* __EMDX__ : ECMWF European Centre for Medium-Range Weather Forecasts [km]
* __EGDX__ : UKMet global model [km]
* __HWDX__ : HWRF Hurricane Weather Research and Forecast system [km]
    * Nested Grid point (13.5-4.5-1.5km)

#### Difference (north-south) between the model predicted track and the track model consensus.
* __OBDY__ : north-south difference [km] of the observed storm position minus that of the track model consensus. _This is what we are trying to predict for the east-west displacement._
* __AVDY__ : GFS Global Forecast System (FV3-GFS) [km]
    * Finite Volume Cube Sphere (~13km)
* __EMDY__ : ECMWF European Centre for Medium-Range Weather Forecasts [km]
* __EGDY__ : UKMet global model [km]
* __HWDY__ : HWRF Hurricane Weather Research and Forecast system [km]
    * Nested Grid point (13.5-4.5-1.5km)

## Notes
* All floating point data given in .1f format.

## References
* https://www.nhc.noaa.gov/modelsummary.shtml
* https://www.ecmwf.int/en/forecasts/charts/latest-tropical-cyclones-forecast
* https://yaleclimateconnections.org/2020/08/the-most-reliable-hurricane-models-according-to-their-2019-performance/