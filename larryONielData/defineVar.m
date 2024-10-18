function varid = defineVar(ncid, varname, vartype, dimidList, varUnits, varLongName)
    varid = netcdf.defVar(ncid, varname, vartype, dimidList);
    netcdf.putAtt(ncid,varid,'units',varUnits)
    netcdf.putAtt(ncid,varid,'long_name', varLongName)
end