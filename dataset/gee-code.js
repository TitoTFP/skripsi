/**** ==============================
 *  ROI dari Cloud Assets SHP
 *  ============================== ****/

// Ganti ini kalau Asset ID project-mu berbeda.
// Cara cek: klik asset di tab Assets → copy Asset ID.
var ASSET_ROOT = 'projects/project-0ff5177f-3bd2-4142-a96/assets';

// Mapping nama tampilan UI -> Asset ID SHP
var roiAssets = {
  'Aceh Besar': ASSET_ROOT + '/Kabupaten_Aceh_Besar_KAB_KOTA',
  'Aceh Tamiang': ASSET_ROOT + '/Kabupaten_Aceh_Tamiang_KAB_KOTA',
  'Aceh Timur': ASSET_ROOT + '/Kabupaten_Aceh_Timur_KAB_KOTA',
  'Aceh Utara': ASSET_ROOT + '/Kabupaten_Aceh_Utara_KAB_KOTA',
  'Agam': ASSET_ROOT + '/Kabupaten_Agam_KAB_KOTA',
  'Bireuen': ASSET_ROOT + '/Kabupaten_Bireuen_KAB_KOTA',
  'Pasaman Barat': ASSET_ROOT + '/Kabupaten_Pasaman_Barat_KAB_KOTA',
  'Pidie Jaya': ASSET_ROOT + '/Kabupaten_Pidie_Jaya_KAB_KOTA',
  'Pidie': ASSET_ROOT + '/Kabupaten_Pidie_KAB_KOTA',
  'Kota Banda Aceh': ASSET_ROOT + '/Kota_Banda_Aceh_KAB_KOTA',
  'Kota Langsa': ASSET_ROOT + '/Kota_Langsa_KAB_KOTA'
};

// List nama wilayah untuk dropdown UI
var namaWilayah = Object.keys(roiAssets);

// Fungsi ambil ROI dari asset
function getROIFromAsset(nama) {
  return ee.FeatureCollection(roiAssets[nama]);
}

// Optional: gabungan semua ROI, kalau nanti butuh ditampilkan sekaligus
var semuaROI = ee.FeatureCollection([]);

namaWilayah.forEach(function(nama) {
  var fc = getROIFromAsset(nama).map(function(f) {
    return f.set('roi_name', nama);
  });

  semuaROI = semuaROI.merge(fc);
});


/**** ==============================
 *  Variabel global
 *  ============================== ****/

var currentCollections = {
  s2: null,
  s1: null
};

var currentROI = null;
var currentROIGeom = null;
var currentCRS = null;
var currentCRSInfo = null;

var sceneDict = {
  s2: {},
  s1: {}
};

var sceneCheckboxes = {
  s2: {},
  s1: {}
};

var sceneListStatus = {
  s2: 'not_loaded',
  s1: 'not_loaded'
};


/**** ==============================
 *  Fungsi masking Sentinel-2
 *  ============================== ****/

function maskS2Clouds(image) {
  var scl = image.select('SCL');

  var mask = scl.neq(3)     // cloud shadow
    .and(scl.neq(8))        // cloud medium probability
    .and(scl.neq(9))        // cloud high probability
    .and(scl.neq(10))       // cirrus
    .and(scl.neq(11));      // snow / ice

  return image.updateMask(mask)
    .divide(10000)
    .copyProperties(image, image.propertyNames());
}

function scaleS2Reflectance(image) {
  return image.divide(10000)
    .copyProperties(image, image.propertyNames());
}


/**** ==============================
 *  UI Panel
 *  ============================== ****/

var panel = ui.Panel({
  style: {
    width: '410px',
    padding: '12px'
  }
});

var title = ui.Label({
  value: 'Pilih Citra Sentinel',
  style: {
    fontSize: '20px',
    fontWeight: 'bold',
    margin: '0 0 10px 0'
  }
});

var wilayahSelect = ui.Select({
  items: namaWilayah,
  value: 'Aceh Utara',
  placeholder: 'Pilih wilayah',
  style: {
    stretch: 'horizontal'
  }
});

var sentinel2Checkbox = ui.Checkbox({
  label: 'Sentinel-2',
  value: true
});

var sentinel1Checkbox = ui.Checkbox({
  label: 'Sentinel-1',
  value: false
});

var sensorPanel = ui.Panel({
  widgets: [
    sentinel2Checkbox,
    sentinel1Checkbox
  ],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    stretch: 'horizontal'
  }
});

var modeSelect = ui.Select({
  items: ['Komposit median', 'Scene tunggal'],
  value: 'Komposit median',
  placeholder: 'Pilih mode tampilan',
  style: {
    stretch: 'horizontal'
  }
});

var startDateBox = ui.Textbox({
  placeholder: 'YYYY-MM-DD',
  value: '2025-11-01',
  style: {
    stretch: 'horizontal'
  }
});

var endDateBox = ui.Textbox({
  placeholder: 'YYYY-MM-DD',
  value: '2025-11-30',
  style: {
    stretch: 'horizontal'
  }
});

var cloudSlider = ui.Slider({
  min: 0,
  max: 100,
  value: 30,
  step: 5,
  style: {
    stretch: 'horizontal'
  }
});

var s2CloudMaskCheckbox = ui.Checkbox({
  label: 'Aktifkan penghilang awan Sentinel-2',
  value: true,
  style: {
    stretch: 'horizontal'
  }
});

var polarizationSelect = ui.Select({
  items: ['BOTH', 'VV', 'VH'],
  value: 'BOTH',
  style: {
    stretch: 'horizontal'
  }
});

var orbitSelect = ui.Select({
  items: ['BOTH', 'ASCENDING', 'DESCENDING'],
  value: 'BOTH',
  style: {
    stretch: 'horizontal'
  }
});

var searchButton = ui.Button({
  label: 'Cari Citra',
  style: {
    stretch: 'horizontal'
  }
});

var s2SceneSelect = ui.Select({
  items: [],
  placeholder: 'Scene Sentinel-2 akan muncul setelah Cari Citra',
  style: {
    stretch: 'horizontal'
  }
});

var s1SceneSelect = ui.Select({
  items: [],
  placeholder: 'Scene Sentinel-1 akan muncul setelah Cari Citra',
  style: {
    stretch: 'horizontal'
  }
});

var s2CompositeScenePanel = ui.Panel({
  style: {
    stretch: 'horizontal'
  }
});

var s1CompositeScenePanel = ui.Panel({
  style: {
    stretch: 'horizontal'
  }
});

var showButton = ui.Button({
  label: 'Tampilkan',
  style: {
    stretch: 'horizontal'
  }
});

var downloadButton = ui.Button({
  label: 'Download / Export ke Google Drive',
  style: {
    stretch: 'horizontal'
  }
});

var infoLabel = ui.Label({
  value: 'Siap.',
  style: {
    margin: '10px 0 0 0',
    whiteSpace: 'pre'
  }
});


/**** ==============================
 *  Susun UI
 *  ============================== ****/

panel.add(title);

panel.add(ui.Label('1. Pilih wilayah'));
panel.add(wilayahSelect);

panel.add(ui.Label('2. Pilih sensor'));
panel.add(sensorPanel);

panel.add(ui.Label('3. Pilih mode tampilan'));
panel.add(modeSelect);

panel.add(ui.Label('4. Tanggal awal'));
panel.add(startDateBox);

panel.add(ui.Label('5. Tanggal akhir'));
panel.add(endDateBox);

panel.add(ui.Label('6. Max cloud Sentinel-2 (%)'));
panel.add(cloudSlider);

panel.add(ui.Label('6B. Penghilang awan Sentinel-2'));
panel.add(s2CloudMaskCheckbox);

panel.add(ui.Label('7. Polarization Sentinel-1'));
panel.add(polarizationSelect);

panel.add(ui.Label('8. Orbit Sentinel-1'));
panel.add(orbitSelect);

panel.add(searchButton);

panel.add(ui.Label('9A. Pilih scene tunggal Sentinel-2'));
panel.add(s2SceneSelect);
panel.add(ui.Label('9A-2. Scene Sentinel-2 untuk komposit median'));
panel.add(s2CompositeScenePanel);


panel.add(ui.Label('9B. Pilih scene tunggal Sentinel-1'));
panel.add(s1SceneSelect);
panel.add(ui.Label('9B-2. Scene Sentinel-1 untuk komposit median'));
panel.add(s1CompositeScenePanel);


panel.add(showButton);
panel.add(downloadButton);
panel.add(infoLabel);

ui.root.insert(0, panel);


/**** ==============================
 *  Fungsi utilitas
 *  ============================== ****/

function cleanName(text) {
  return text
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_]/g, '');
}

function getCompositeDateSuffix() {
  return 'periode_' + startDateBox.getValue() + '_sampai_' + endDateBox.getValue();
}

function getAutoUtmCrsInfo(geom) {
  var centroid = geom.centroid(1);
  var coords = centroid.coordinates();

  var lon = ee.Number(coords.get(0));
  var lat = ee.Number(coords.get(1));

  // Rumus zona UTM:
  // zone = floor((lon + 180) / 6) + 1
  var zone = lon
    .add(180)
    .divide(6)
    .floor()
    .add(1)
    .int();

  // EPSG:
  // 326xx = UTM Northern Hemisphere
  // 327xx = UTM Southern Hemisphere
  var epsgCode = ee.Number(
    ee.Algorithms.If(
      lat.gte(0),
      ee.Number(32600).add(zone),
      ee.Number(32700).add(zone)
    )
  );

  var crs = ee.String('EPSG:').cat(epsgCode.format('%d'));

  return ee.Dictionary({
    crs: crs,
    epsg: epsgCode,
    zone: zone,
    lon: lon,
    lat: lat
  });
}

function getDateRange() {
  var start = startDateBox.getValue();
  var end = endDateBox.getValue();

  return {
    start: ee.Date(start),

    // filterDate end date bersifat exclusive.
    // Jadi endDate + 1 hari agar tanggal akhir ikut masuk.
    end: ee.Date(end).advance(1, 'day')
  };
}

function getSelectedSensors() {
  return {
    s2: sentinel2Checkbox.getValue(),
    s1: sentinel1Checkbox.getValue()
  };
}

function getSelectedPolarizations() {
  var pol = polarizationSelect.getValue();

  if (pol === 'BOTH') {
    return ['VV', 'VH'];
  }

  return [pol];
}

function resetSceneDropdowns() {
  s2SceneSelect.items().reset([]);
  s2SceneSelect.setPlaceholder('Scene Sentinel-2 akan muncul setelah Cari Citra');

  s1SceneSelect.items().reset([]);
  s1SceneSelect.setPlaceholder('Scene Sentinel-1 akan muncul setelah Cari Citra');

  sceneDict = {
    s2: {},
    s1: {}
  };

  sceneCheckboxes = {
    s2: {},
    s1: {}
  };

  sceneListStatus = {
    s2: 'not_loaded',
    s1: 'not_loaded'
  };

  s2CompositeScenePanel.clear();
  s1CompositeScenePanel.clear();
}

function fillCompositeSceneCheckboxes(sensorKey, targetPanel, labelList) {
  targetPanel.clear();
  sceneCheckboxes[sensorKey] = {};
  sceneListStatus[sensorKey] = 'loading';

  if (!labelList || labelList.length === 0) {
    sceneListStatus[sensorKey] = 'empty';
    targetPanel.add(ui.Label('Tidak ada scene untuk komposit median.'));
    return;
  }

  targetPanel.add(ui.Label('Centang scene yang dipakai untuk komposit median:'));
  sceneListStatus[sensorKey] = 'loaded';

  for (var i = 0; i < labelList.length; i++) {
    var label = labelList[i];
    var checkbox = ui.Checkbox({
      label: label,
      value: true,
      style: {
        stretch: 'horizontal'
      }
    });

    sceneCheckboxes[sensorKey][label] = checkbox;
    targetPanel.add(checkbox);
  }
}

function getCheckedSceneLabels(sensorKey) {
  var labels = [];
  var checkboxes = sceneCheckboxes[sensorKey];

  for (var label in checkboxes) {
    if (checkboxes.hasOwnProperty(label) && checkboxes[label].getValue()) {
      labels.push(label);
    }
  }

  return labels;
}

function getCheckedSceneIndexes(sensorKey) {
  var labels = getCheckedSceneLabels(sensorKey);
  var indexes = [];

  for (var i = 0; i < labels.length; i++) {
    var scene = sceneDict[sensorKey][labels[i]];

    if (scene) {
      indexes.push(scene.index);
    }
  }

  return indexes;
}

function getCompositeCollection(sensorKey, collection) {
  var status = sceneListStatus[sensorKey];
  var selectableCount = Object.keys(sceneCheckboxes[sensorKey]).length;

  if (status === 'empty') {
    return null;
  }

  if (selectableCount === 0) {
    return {
      collection: collection,
      count: null
    };
  }

  var indexes = getCheckedSceneIndexes(sensorKey);

  if (indexes.length === 0) {
    return null;
  }

  return {
    collection: collection.filter(ee.Filter.inList('system:index', indexes)),
    count: indexes.length
  };
}

function getCompositeNameSuffix(sensorKey) {
  var checkedCount = getCheckedSceneLabels(sensorKey).length;

  if (checkedCount === 0) {
    return getCompositeDateSuffix();
  }

  return getCompositeDateSuffix() + '_' + checkedCount + 'scene';
}

function validateCompositeSceneSelection(sensorKey, sensorName) {
  var status = sceneListStatus[sensorKey];
  var selectableCount = Object.keys(sceneCheckboxes[sensorKey]).length;

  if (status === 'empty') {
    infoLabel.setValue('Tidak ada scene ' + sensorName + ' untuk komposit median.');
    print('Tidak ada scene ' + sensorName + ' untuk komposit median.');
    return false;
  }

  if (selectableCount > 0 && getCheckedSceneLabels(sensorKey).length === 0) {
    infoLabel.setValue(
      'Centang minimal satu scene ' + sensorName +
      ' untuk komposit median.'
    );
    print('Tidak ada scene ' + sensorName + ' yang dicentang untuk komposit median.');
    return false;
  }

  return true;
}


/**** ==============================
 *  Fungsi buat koleksi Sentinel-2
 *  ============================== ****/

function buildS2Collection(roiGeom, dates) {
  var maxCloud = cloudSlider.getValue();
  var useCloudMask = s2CloudMaskCheckbox.getValue();

  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roiGeom)
    .filterDate(dates.start, dates.end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxCloud));

  if (useCloudMask) {
    s2 = s2.map(maskS2Clouds);
  } else {
    s2 = s2.map(scaleS2Reflectance);
  }

  return s2;
}


/**** ==============================
 *  Fungsi buat koleksi Sentinel-1
 *  ============================== ****/

function buildS1Collection(roiGeom, dates) {
  var pols = getSelectedPolarizations();
  var orbit = orbitSelect.getValue();

  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roiGeom)
    .filterDate(dates.start, dates.end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10));

  // Kalau BOTH, pastikan image punya VV dan VH.
  // Kalau hanya VV/VH, cukup filter polarization tersebut.
  for (var i = 0; i < pols.length; i++) {
    s1 = s1.filter(
      ee.Filter.listContains('transmitterReceiverPolarisation', pols[i])
    );
  }

  if (orbit !== 'BOTH') {
    s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', orbit));
  }

  return s1;
}


/**** ==============================
 *  Fungsi isi dropdown scene Sentinel-2
 *  ============================== ****/

function fillS2SceneDropdown(collection) {
  s2SceneSelect.items().reset([]);
  s2SceneSelect.setPlaceholder('Memuat scene Sentinel-2...');

  var sceneInfo = collection.map(function(img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var index = ee.String(img.get('system:index'));
    var cloud = ee.Number(img.get('CLOUDY_PIXEL_PERCENTAGE')).format('%.1f');

    var label = date
      .cat(' | cloud: ')
      .cat(cloud)
      .cat('% | ')
      .cat(index);

    return ee.Feature(null, {
      label: label,
      index: index,
      date: date
    });
  });

  var labels = sceneInfo.aggregate_array('label');
  var indexes = sceneInfo.aggregate_array('index');
  var dates = sceneInfo.aggregate_array('date');

  labels.evaluate(function(labelList) {
    indexes.evaluate(function(indexList) {
      dates.evaluate(function(dateList) {

        if (!labelList || labelList.length === 0) {
          s2SceneSelect.items().reset([]);
          s2SceneSelect.setPlaceholder('Tidak ada scene Sentinel-2');
          fillCompositeSceneCheckboxes('s2', s2CompositeScenePanel, []);
          return;
        }

        sceneDict.s2 = {};

        for (var i = 0; i < labelList.length; i++) {
          sceneDict.s2[labelList[i]] = {
            index: indexList[i],
            date: dateList[i]
          };
        }

        s2SceneSelect.items().reset(labelList);
        s2SceneSelect.setValue(labelList[0]);
        fillCompositeSceneCheckboxes('s2', s2CompositeScenePanel, labelList);
      });
    });
  });
}


/**** ==============================
 *  Fungsi isi dropdown scene Sentinel-1
 *  ============================== ****/

function fillS1SceneDropdown(collection) {
  s1SceneSelect.items().reset([]);
  s1SceneSelect.setPlaceholder('Memuat scene Sentinel-1...');

  var sceneInfo = collection.map(function(img) {
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var index = ee.String(img.get('system:index'));
    var orbit = ee.String(img.get('orbitProperties_pass'));

    var label = date
      .cat(' | ')
      .cat(orbit)
      .cat(' | ')
      .cat(index);

    return ee.Feature(null, {
      label: label,
      index: index,
      date: date
    });
  });

  var labels = sceneInfo.aggregate_array('label');
  var indexes = sceneInfo.aggregate_array('index');
  var dates = sceneInfo.aggregate_array('date');

  labels.evaluate(function(labelList) {
    indexes.evaluate(function(indexList) {
      dates.evaluate(function(dateList) {

        if (!labelList || labelList.length === 0) {
          s1SceneSelect.items().reset([]);
          s1SceneSelect.setPlaceholder('Tidak ada scene Sentinel-1');
          fillCompositeSceneCheckboxes('s1', s1CompositeScenePanel, []);
          return;
        }

        sceneDict.s1 = {};

        for (var i = 0; i < labelList.length; i++) {
          sceneDict.s1[labelList[i]] = {
            index: indexList[i],
            date: dateList[i]
          };
        }

        s1SceneSelect.items().reset(labelList);
        s1SceneSelect.setValue(labelList[0]);
        fillCompositeSceneCheckboxes('s1', s1CompositeScenePanel, labelList);
      });
    });
  });
}


/**** ==============================
 *  Fungsi cari citra
 *  ============================== ****/

function cariCitra() {
  Map.layers().reset();
  resetSceneDropdowns();

  var wilayah = wilayahSelect.getValue();
  var mode = modeSelect.getValue();
  var dates = getDateRange();
  var selectedSensors = getSelectedSensors();

  if (!selectedSensors.s2 && !selectedSensors.s1) {
    infoLabel.setValue('Pilih minimal satu sensor: Sentinel-2 atau Sentinel-1.');
    return;
  }

  if (!roiAssets[wilayah]) {
    infoLabel.setValue('Asset ROI untuk wilayah ini belum tersedia: ' + wilayah);
    print('Asset ROI belum tersedia:', wilayah);
    return;
  }
  
  var selectedROI = getROIFromAsset(wilayah);
  
  // Dissolve geometry agar semua polygon dalam SHP dianggap satu ROI
  var roiGeom = selectedROI.geometry().dissolve(1);

  currentROI = selectedROI;
  currentROIGeom = roiGeom;
  currentCRS = null;
  currentCRSInfo = null;
  
  // Hitung CRS UTM otomatis berdasarkan centroid ROI.
  // Hasilnya di-evaluate agar menjadi string client-side,
  // karena Export.image.toDrive membutuhkan crs berupa string biasa.
  getAutoUtmCrsInfo(roiGeom).evaluate(function(info) {
    currentCRSInfo = info;
    currentCRS = info.crs;
  
    print('CRS UTM otomatis:', currentCRS);
    print('Info CRS UTM:', info);
  
    infoLabel.setValue(
      'Wilayah: ' + wilayah +
      '\nMode: ' + mode +
      '\nCRS UTM otomatis: ' + currentCRS +
      '\nZona UTM: ' + info.zone +
      '\nCentroid lon: ' + info.lon.toFixed(6) +
      '\nCentroid lat: ' + info.lat.toFixed(6) +
      '\n\nMencari citra...'
    );
  });
  
  currentCollections = {
    s2: null,
    s1: null
  };

  Map.centerObject(currentROI, 9);
  Map.addLayer(
    currentROI,
    {color: 'red'},
    'ROI - ' + wilayah
  );

  var infoText = 'Wilayah: ' + wilayah +
    '\nMode: ' + mode +
    '\nMencari citra...';

  infoLabel.setValue(infoText);

  print('Wilayah:', wilayah);
  print('Mode:', mode);
  print('Tanggal awal:', dates.start);
  print('Tanggal akhir exclusive:', dates.end);

  if (selectedSensors.s2) {
    currentCollections.s2 = buildS2Collection(roiGeom, dates);

    print('Sentinel-2 collection:', currentCollections.s2);
    print('Jumlah citra Sentinel-2:', currentCollections.s2.size());

    fillS2SceneDropdown(currentCollections.s2);
  }

  if (selectedSensors.s1) {
    currentCollections.s1 = buildS1Collection(roiGeom, dates);

    print('Sentinel-1 collection:', currentCollections.s1);
    print('Jumlah citra Sentinel-1:', currentCollections.s1.size());

    fillS1SceneDropdown(currentCollections.s1);
  }

  var s2Count = selectedSensors.s2
    ? currentCollections.s2.size()
    : ee.Number(0);

  var s1Count = selectedSensors.s1
    ? currentCollections.s1.size()
    : ee.Number(0);

  ee.Dictionary({
    s2: s2Count,
    s1: s1Count
  }).evaluate(function(result) {
    var text = 'Citra ditemukan:' +
      '\nSentinel-2: ' + result.s2 +
      '\nSentinel-1: ' + result.s1 +
      '\n\nMode aktif: ' + mode;

    if (mode === 'Scene tunggal') {
      text += '\nPilih scene pada dropdown lalu klik Tampilkan.';
    } else {
      text += '\nSemua scene dicentang otomatis untuk komposit median.';
      text += '\nHapus centang scene yang tidak ingin dipakai.';
    }

    infoLabel.setValue(text);
  });
}


/**** ==============================
 *  Fungsi tampilkan Sentinel-2
 *  ============================== ****/

function tampilkanS2(mode, wilayah) {
  if (currentCollections.s2 === null) {
    return;
  }

  if (mode === 'Komposit median') {
    var s2Composite = getCompositeCollection('s2', currentCollections.s2);

    if (s2Composite === null) {
      infoLabel.setValue('Centang minimal satu scene Sentinel-2 untuk komposit median.');
      print('Tidak ada scene Sentinel-2 yang dicentang untuk komposit median.');
      return;
    }

    var s2Median = s2Composite.collection
      .median()
      .clip(currentROIGeom);

    var countText = s2Composite.count === null
      ? ''
      : ' (' + s2Composite.count + ' scene)';

    Map.addLayer(
      s2Median,
      {
        bands: ['B4', 'B3', 'B2'],
        min: 0,
        max: 0.3,
        gamma: 1.2
      },
      'Sentinel-2 Median RGB - ' + wilayah + countText
    );

  } else {
    var selectedLabel = s2SceneSelect.getValue();

    if (!selectedLabel || !sceneDict.s2[selectedLabel]) {
      print('Scene Sentinel-2 belum dipilih.');
      return;
    }

    var selectedIndex = sceneDict.s2[selectedLabel].index;
    var selectedDate = sceneDict.s2[selectedLabel].date;

    var s2Scene = ee.Image(
      currentCollections.s2
        .filter(ee.Filter.eq('system:index', selectedIndex))
        .first()
    ).clip(currentROIGeom);

    Map.addLayer(
      s2Scene,
      {
        bands: ['B4', 'B3', 'B2'],
        min: 0,
        max: 0.3,
        gamma: 1.2
      },
      'Sentinel-2 Scene RGB - ' + selectedDate
    );
  }
}


/**** ==============================
 *  Fungsi tampilkan Sentinel-1
 *  ============================== ****/

function tampilkanS1(mode, wilayah) {
  if (currentCollections.s1 === null) {
    return;
  }

  var pols = getSelectedPolarizations();

  if (mode === 'Komposit median') {
    var s1Composite = getCompositeCollection('s1', currentCollections.s1);

    if (s1Composite === null) {
      infoLabel.setValue('Centang minimal satu scene Sentinel-1 untuk komposit median.');
      print('Tidak ada scene Sentinel-1 yang dicentang untuk komposit median.');
      return;
    }

    var countText = s1Composite.count === null
      ? ''
      : ' (' + s1Composite.count + ' scene)';

    for (var i = 0; i < pols.length; i++) {
      var pol = pols[i];

      var s1Median = s1Composite.collection
        .select(pol)
        .median()
        .clip(currentROIGeom);

      Map.addLayer(
        s1Median,
        {
          min: -25,
          max: 0
        },
        'Sentinel-1 Median ' + pol + ' - ' + wilayah + countText
      );
    }
  } else {
    var selectedLabel = s1SceneSelect.getValue();

    if (!selectedLabel || !sceneDict.s1[selectedLabel]) {
      print('Scene Sentinel-1 belum dipilih.');
      return;
    }

    var selectedIndex = sceneDict.s1[selectedLabel].index;
    var selectedDate = sceneDict.s1[selectedLabel].date;

    var s1Image = ee.Image(
      currentCollections.s1
        .filter(ee.Filter.eq('system:index', selectedIndex))
        .first()
    ).clip(currentROIGeom);

    for (var j = 0; j < pols.length; j++) {
      var polScene = pols[j];

      Map.addLayer(
        s1Image.select(polScene),
        {
          min: -25,
          max: 0
        },
        'Sentinel-1 Scene ' + polScene + ' - ' + selectedDate
      );
    }
  }
}


/**** ==============================
 *  Fungsi tampilkan citra
 *  ============================== ****/

function tampilkanCitra() {
  if (currentROI === null) {
    infoLabel.setValue('Klik "Cari Citra" dulu.');
    return;
  }

  Map.layers().reset();

  var wilayah = wilayahSelect.getValue();
  var mode = modeSelect.getValue();
  var selectedSensors = getSelectedSensors();

  if (mode === 'Komposit median') {
    if (selectedSensors.s2 && !validateCompositeSceneSelection('s2', 'Sentinel-2')) {
      return;
    }

    if (selectedSensors.s1 && !validateCompositeSceneSelection('s1', 'Sentinel-1')) {
      return;
    }
  }

  Map.centerObject(currentROI, 9);
  Map.addLayer(
    currentROI,
    {color: 'red'},
    'ROI - ' + wilayah
  );

  if (selectedSensors.s2) {
    tampilkanS2(mode, wilayah);
  }

  if (selectedSensors.s1) {
    tampilkanS1(mode, wilayah);
  }

  var sensorText = [];

  if (selectedSensors.s2) {
    sensorText.push('Sentinel-2');
  }

  if (selectedSensors.s1) {
    sensorText.push('Sentinel-1');
  }

  infoLabel.setValue(
    'Wilayah: ' + wilayah +
    '\nSensor: ' + sensorText.join(', ') +
    '\nMode: ' + mode +
    '\nCRS UTM otomatis: ' + currentCRS +
    '\nCloud mask S2: ' + (s2CloudMaskCheckbox.getValue() ? 'Aktif' : 'Nonaktif') +
    '\nPolarization S1: ' + polarizationSelect.getValue() +
    '\nOrbit S1: ' + orbitSelect.getValue()
  );
}

function exportS2(mode, wilayah) {
  if (currentCollections.s2 === null) {
    print('Tidak ada koleksi Sentinel-2 untuk diexport.');
    return;
  }

  var image;
  var nameSuffix;

  if (mode === 'Komposit median') {
    var s2Composite = getCompositeCollection('s2', currentCollections.s2);

    if (s2Composite === null) {
      infoLabel.setValue('Centang minimal satu scene Sentinel-2 sebelum export komposit median.');
      print('Tidak ada scene Sentinel-2 yang dicentang untuk export komposit median.');
      return;
    }

    image = s2Composite.collection
      .median()
      .clip(currentROIGeom);

    nameSuffix = getCompositeNameSuffix('s2');

  } else {
    var selectedLabel = s2SceneSelect.getValue();

    if (!selectedLabel || !sceneDict.s2[selectedLabel]) {
      print('Pilih scene Sentinel-2 dulu sebelum export.');
      return;
    }

    var selectedIndex = sceneDict.s2[selectedLabel].index;
    var selectedDate = sceneDict.s2[selectedLabel].date;

    image = ee.Image(
      currentCollections.s2
        .filter(ee.Filter.eq('system:index', selectedIndex))
        .first()
    ).clip(currentROIGeom);

    nameSuffix = selectedDate;
  }

  // Band umum Sentinel-2 untuk analisis
  var exportImage = image.select([
    'B2',   // Blue
    'B3',   // Green
    'B4',   // Red
    'B8',   // NIR
    'B11',  // SWIR 1
    'B12'   // SWIR 2
  ]);

  var fileName = cleanName(
    'S2_' + wilayah + '_' + mode + '_' + nameSuffix
  );
  
  if (currentCRS === null) {
  infoLabel.setValue(
    'CRS UTM otomatis belum siap.\n' +
    'Klik "Cari Citra" dulu dan tunggu sampai CRS muncul.'
  );
  print('CRS UTM otomatis belum siap.');
  return;
  }

  Export.image.toDrive({
    image: exportImage,
    description: fileName,
    folder: 'GEE_Exports',
    fileNamePrefix: fileName,
    region: currentROIGeom,
    scale: 10,
    crs: currentCRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });

  print('Task export Sentinel-2 dibuat:', fileName);
}

function exportS1(mode, wilayah) {
  if (currentCollections.s1 === null) {
    print('Tidak ada koleksi Sentinel-1 untuk diexport.');
    return;
  }

  var pols = getSelectedPolarizations();

  var image;
  var nameSuffix;

  if (mode === 'Komposit median') {
    var s1Composite = getCompositeCollection('s1', currentCollections.s1);

    if (s1Composite === null) {
      infoLabel.setValue('Centang minimal satu scene Sentinel-1 sebelum export komposit median.');
      print('Tidak ada scene Sentinel-1 yang dicentang untuk export komposit median.');
      return;
    }

    image = s1Composite.collection
      .select(pols)
      .median()
      .clip(currentROIGeom);

    nameSuffix = getCompositeNameSuffix('s1');

  } else {
    var selectedLabel = s1SceneSelect.getValue();

    if (!selectedLabel || !sceneDict.s1[selectedLabel]) {
      print('Pilih scene Sentinel-1 dulu sebelum export.');
      return;
    }

    var selectedIndex = sceneDict.s1[selectedLabel].index;
    var selectedDate = sceneDict.s1[selectedLabel].date;

    image = ee.Image(
      currentCollections.s1
        .filter(ee.Filter.eq('system:index', selectedIndex))
        .first()
    )
      .select(pols)
      .clip(currentROIGeom);

    nameSuffix = selectedDate;
  }

  var fileName = cleanName(
    'S1_' + wilayah + '_' + mode + '_' + polarizationSelect.getValue() + '_' + orbitSelect.getValue() + '_' + nameSuffix
  );
  
  if (currentCRS === null) {
  infoLabel.setValue(
    'CRS UTM otomatis belum siap.\n' +
    'Klik "Cari Citra" dulu dan tunggu sampai CRS muncul.'
  );
  print('CRS UTM otomatis belum siap.');
  return;
  }

  Export.image.toDrive({
    image: image,
    description: fileName,
    folder: 'GEE_Exports',
    fileNamePrefix: fileName,
    region: currentROIGeom,
    scale: 10,
    crs: currentCRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });

  print('Task export Sentinel-1 dibuat:', fileName);
}

function downloadCitra() {
  if (currentROI === null) {
    infoLabel.setValue('Klik "Cari Citra" dulu sebelum download.');
    return;
  }

  var wilayah = wilayahSelect.getValue();
  var mode = modeSelect.getValue();
  var selectedSensors = getSelectedSensors();

  if (!selectedSensors.s2 && !selectedSensors.s1) {
    infoLabel.setValue('Pilih minimal satu sensor sebelum download.');
    return;
  }

  if (mode === 'Komposit median') {
    if (selectedSensors.s2 && !validateCompositeSceneSelection('s2', 'Sentinel-2')) {
      return;
    }

    if (selectedSensors.s1 && !validateCompositeSceneSelection('s1', 'Sentinel-1')) {
      return;
    }
  }

  if (selectedSensors.s2) {
    exportS2(mode, wilayah);
  }

  if (selectedSensors.s1) {
    exportS1(mode, wilayah);
  }

  infoLabel.setValue(
    'Task export sudah dibuat.\n' +
    'Buka tab Tasks di kanan atas GEE,\n' +
    'lalu klik Run pada task export.'
  );
}

/**** ==============================
 *  Event tombol
 *  ============================== ****/

searchButton.onClick(cariCitra);
showButton.onClick(tampilkanCitra);
downloadButton.onClick(downloadCitra);

// Jalankan default saat script dibuka
cariCitra();
