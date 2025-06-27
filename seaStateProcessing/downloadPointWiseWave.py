import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i",
  variables=["VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW", "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW", "VPED", "VSDX", "VSDY", "VTM01_SW1", "VTM01_SW2", "VTM01_WW", "VTM02", "VTM10", "VTPK"],
  minimum_longitude=179,
  maximum_longitude=181,
  minimum_latitude=-1,
  maximum_latitude=1,
  start_datetime="1999-01-01T00:00:00",
  end_datetime="2023-04-30T21:00:00",
)
