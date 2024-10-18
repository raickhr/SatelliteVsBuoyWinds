
ds = load('../../downloads/larryNielData/larry2020/EXP3/qscat_ascata_TAO_collocations_mar20_exp3.mat');

nBuoy = length(ds.ASCATA_TAO_collocated);
% buoy_names = strings{nBuoy};
for i = 1:nBuoy
    buoy_name = ds.ASCATA_TAO_collocated{1,i}.buoy_name;
    lat = ds.ASCATA_TAO_collocated{1,i}.lat;
    lon = ds.ASCATA_TAO_collocated{1,i}.lon;
    timeArr = ds.ASCATA_TAO_collocated{1,i}.ascata.time;

    t = datetime(timeArr, 'ConvertFrom', 'datenum');
    timeArr = exceltime(t, '1904');
    tlen = length(timeArr);

    if tlen == 0
        continue
    end

    disp(buoy_name)
    disp(length(timeArr))


    satWspd = ds.ASCATA_TAO_collocated{1,i}.ascata.sat_wspd10n;
    satUwind = ds.ASCATA_TAO_collocated{1,i}.ascata.sat_u10n;
    satVwind = ds.ASCATA_TAO_collocated{1,i}.ascata.sat_v10n;

    buoyWspd = ds.ASCATA_TAO_collocated{1,i}.ascata.buoy_wspd10n;
    buoyUwind = ds.ASCATA_TAO_collocated{1,i}.ascata.buoy_u10n;
    buoyVwind = ds.ASCATA_TAO_collocated{1,i}.ascata.buoy_v10n;

    % buoysst = ds.ASCATA_TAO_collocated{1,i}.ascata.buoy_sst;
    % buoyairt = ds.ASCATA_TAO_collocated{1,i}.ascata.buoy_atmp;



    fileName = sprintf('../../downloads/larryNielData/larry2020/EXP3/fromLarry_%s_ASCATdata.nc',buoy_name);
    ncid = netcdf.create(fileName,'CLOBBER');
    netcdf.putAtt(ncid,netcdf.getConstant('NC_GLOBAL'),'Description', "Netcdf file created from Larry's matchup")

    dimid = netcdf.defDim(ncid,'time',length(timeArr));

    varname = 'time';
    vartype = 'NC_DOUBLE';
    dimidList = dimid;
    varUnits = "days since 1904-01-01 00:00:00";
    varLongName = '';
    timeVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);

    varname = 'sat_wspd10n';
    vartype = 'NC_DOUBLE';
    dimidList = dimid;
    varUnits = "m/s";
    varLongName = 'satellite equivalent neutral wind speed at 10m';
    satWspdVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);

    varname = 'buoy_wspd10n';
    vartype = 'NC_DOUBLE';
    dimidList = dimid;
    varUnits = "m/s";
    varLongName = 'buoy equivaelnt neutral wind speed at 10m';
    buoyWspdVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);

    varname = 'sat_u10n';
    vartype = 'NC_DOUBLE';
    dimidList = dimid;
    varUnits = "m/s";
    varLongName = 'satellite x component of equivalent neutral wind speed at 10m';
    satU10nVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);
   
    varname = 'buoy_u10n';
    vartype = 'NC_DOUBLE';
    dimidList = dimid;
    varUnits = "m/s";
    varLongName = 'buoy x component of equivalent neutral wind speed at 10m';
    buoyU10nVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);

    varname = 'sat_v10n';
    vartype = 'NC_DOUBLE';
    dimidList = dimid;
    varUnits = "m/s";
    varLongName = 'satellite y component of equivalent neutral wind speed at 10m';
    satV10nVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);

    varname = 'buoy_v10n';
    vartype = 'NC_DOUBLE';
    dimidList = dimid;
    varUnits = "m/s";
    varLongName = 'buoy v component of equivalent neutral wind speed at 10m';
    buoyV10nVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % // varname = 'buoy_sst';
    % // vartype = 'NC_DOUBLE';
    % // dimidList = dimid;
    % // varUnits = "deg C";
    % // varLongName = 'buoy SST at 2 m';
    % // buoysstVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);
    % 
    % // varname = 'buoy_airt';
    % // vartype = 'NC_DOUBLE';
    % // dimidList = dimid;
    % // varUnits = "";
    % // varLongName = 'buoy bulk airt at 2 m';
    % // buoyairtVarid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName);

    netcdf.endDef(ncid)
    % data mode

    netcdf.putVar(ncid,timeVarid,timeArr)

    netcdf.putVar(ncid,satWspdVarid,satWspd)
    netcdf.putVar(ncid,satU10nVarid,satUwind)
    netcdf.putVar(ncid,satV10nVarid,satVwind)

    netcdf.putVar(ncid,buoyWspdVarid,buoyWspd)
    netcdf.putVar(ncid,buoyU10nVarid,buoyUwind)
    netcdf.putVar(ncid,buoyV10nVarid,buoyVwind)

    % netcdf.putVar(ncid,buoysstVarid,buoysst)
    % netcdf.putVar(ncid,buoyairtVarid,buoyairt)

    netcdf.close(ncid)
end