
//sections's data
let data = [
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":"1"
  },
  {
    "status":"healthy",
    "prob":0.7,
    "src":"/image/test.jpg",
    "sec":"2"
  },
  {
    "status":"healthy",
    "prob":0.6,
    "src":"/image/rwanda_story.jpeg",
    "sec":"3"
  },
  {
    "status":"a",
    "prob":0.7,
    "src":"/image/test.jpg",
    "sec":"4"
  },
  {
    "status":"healthy",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":"5"
  },
  {
    "status":"healthy",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":"6"
  },
  {
    "status":"healthy",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":"7"
  },
  {
    "status":"healthy",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":"8"
  },
  {
    "status":"a",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":"9"
  },
  {
    "status":"b",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":10
  },
  {
    "status":"c",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":11
  },
  {
    "status":"d",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":12
  },
  {
    "status":"a",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":13
  },
  {
    "status":"b",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":14
  },
  {
    "status":"a",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":15
  },
  {
    "status":"a",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":16
  },
  {
    "status":"d",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":17
  },
  {
    "status":"a",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":18
  },
  {
    "status":"d",
    "prob":0.3,
    "src":"/image/test.jpg",
    "sec":19
  },
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":20
  },
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":21
  },
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":22
  },
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":23
  },
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":24
  },
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":25
  },
  {
    "status":"healthy",
    "prob":0.9,
    "src":"/image/test.jpg",
    "sec":26
  },

] 

//

let last5Rec =[
  {
    "time":"14/5/2023 | 8:15:44",
    "healthy":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "bb":[21,22],
    "bc":[23,24],
    "gm":[25,26],
    "md":[]
  },
  {
    "time":"14/5/2023 | 9:15:44",
  "healthy":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
  "bb":[21,22],
  "bc":[23,24],
  "gm":[25,26],
  "md":[]
  },
  {
    "time":"14/5/2023 | 10:15:44",
    "healthy":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
    "bb":[21,22],
    "bc":[19,20,23,24],
    "gm":[25,26],
    "md":[]
  },
  {
    "time":"14/5/2023 | 11:15:44",
    "healthy":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    "bb":[21,22,17,18],
    "bc":[19,20,23,24],
    "gm":[25,26],
    "md":[]
  },
  {
    "time":"14/5/2023 | 12:15:44",
    "healthy":[1,2,3,4,5,6,7,8,9,14,15,16],
    "bb":[21,22,17,18],
    "bc":[23,24,19,20],
    "gm":[25,26],
    "md":[10,11,12,13,]
  }
]









// let lastRecDate = ['05-14-2023, 12:45', '05/14/2023, 11:45', '05/14/2023, 10:45', '05/14/2023, 9:45',
//       '05/14/2023, 8:45', '05/14/2023, 7:45']

var sectioncount=0,healthycount = 0,infected = 0;   //total numbs of sections, healthy cases, infected cases

var a=0,b=0,c=0,d=0;  // 4 diseases count
var adata=[],bdata=[],cdata=[],ddata=[],hdata=[]  //obj ar for each disease

var hrec=[],arec=[],brec=[],crec=[],drec=[],lastRecDate=[]; // last 5 records info (5 cases n record time)
var datetime;
var recover=[],newa=[],newb=[],newc=[],newd=[];


//take in array of object and then classifi them into different array
function classifiData(data){
  var curh=[],cura=[],curb=[],curc=[],curd=[];
  data.forEach(element => {
    sectioncount+=1
    
    
    switch(element.status){
      case "healthy":
        healthycount+=1;
        hdata.push(element);
        curh.push(parseInt(element.sec));
        break;
      case "a":
        a+=1;
        infected+=1;
        adata.push(element);
        cura.push(parseInt(element.sec));
        break;
      case "b":
        b+=1;
        infected+=1;
        bdata.push(element);
        curb.push(parseInt(element.sec));
        break;
      case "c":
        c+=1;
        infected+=1;
        cdata.push(element);
        curc.push(parseInt(element.sec));
        break;
      case "d":
        d+=1;
        infected+=1;
        ddata.push(element);
        curd.push(parseInt(element.sec));
        break;
      default:
        
        
    }
  });

  last5Rec.forEach(element =>{
    lastRecDate.push(element.time)
    hrec.push(element.healthy.length)
    arec.push(element.bb.length)
    brec.push(element.bc.length)
    crec.push(element.gm.length)
    drec.push(element.md.length)
  });

  var currentdate = new Date(); 
  datetime = currentdate.getDate() + "/"
                + (currentdate.getMonth()+1)  + "/" 
                + currentdate.getFullYear() + " | "  
                + currentdate.getHours() + ":"  
                + currentdate.getMinutes() + ":" 
                + currentdate.getSeconds();

  lastRecDate.push(datetime+" (N)")
  hrec.push(healthycount)
  arec.push(a)
  brec.push(b)
  crec.push(c)
  drec.push(d)

  lastRecDate.reverse();
  hrec.reverse();
  arec.reverse();
  brec.reverse();
  crec.reverse();
  drec.reverse();

  curh.forEach(section =>{
    if (!(section in last5Rec[4].healthy))recover.push(section)
  });
  cura.forEach(section =>{
    if (!(section in last5Rec[4].bb))newa.push(section)
  });
  curb.forEach(section =>{
    if (!(section in last5Rec[4].bc))newb.push(section)
  });
  curc.forEach(section =>{
    if (!(section in last5Rec[4].gm))recover.push(section)
  });
  curd.forEach(section =>{
    if (!(section in last5Rec[4].md))recover.push(section)
  });

  recover.sort()
  newa.sort()
  newb.sort()
  newc.sort()
  newd.sort()

  console.log(curh)
  // console.log(last5Rec[4].healthy)
  console.log("check new")
  console.log(recover)
  console.log(newa)
  console.log(newb)
  console.log(newc)
  console.log(newd)
  
  
  
}


function updateHealthy(){
  document.getElementById("hsum").innerText=`${healthycount} ${healthycount>1?" sections are ":"section is "} healthy `;
  document.getElementById("hl1").innerHTML=`${healthycount} ${healthycount>1?"sections are ":"section is"} in a healthy state`

  hdata.forEach(element=>{
    
    let sid=element.sec.toString()
    var onClickParam = element.src
    
    onClickParam="zoomSection(\""+onClickParam+"\",\""+element.sec+"\")"
    

    document.getElementById("hl2").innerHTML+=`
      <div class="img-block">
        <img id=${sid} src=${element.src} class="s-img" onclick='${onClickParam}'>
        <div class="sec-name">Section ${element.sec}</div>
      </div>
    `

  });

}

// cuz healthy have some more stuff so me separate healthy | 4 disease
function updateSection(secState){
  var plural,secArray,dis;
  
  switch(secState){
    case "k":
      plural=a;
      secArray=adata;
      dis="Bacterial Blight";
      break;
    case "l":
      plural=b;
      secArray=bdata;;
      dis="Brown Streak Disease";
      break;
    case "c":
      plural=c;
      secArray=cdata;;
      dis="Green Mottle";
      break;
    case "d":
      plural=d;
      secArray=ddata;;
      dis="Mosaic Disease";
      break;
    default:
      break;
  }

  

  document.getElementById(secState+"1").innerHTML=`${plural} ${plural>1?"sections are ":"section is"} infected with ${dis}`

  

  secArray.forEach(element=>{
    // console.log(element)

    var onClickParam = element.src
    
    onClickParam="zoomSection(\""+onClickParam+"\",\""+element.sec+"\")"

    document.getElementById(secState+"2").innerHTML+=`
      <div class="img-block">
        <img src=${element.src} class="s-img" onclick='${onClickParam}'>
        <div class="sec-name">Section ${element.sec}</div>
      </div>
    `
  });
}


//show summary as well as section status
function showStatus(data){

  // console.log(healthycount)
  if (healthycount!==0){
    document.getElementById("hl").style.display="flex"
    updateHealthy()
    changeDisplay("hl")
  }
  else{
    hideEmptyCb("hl")
    document.getElementById("hl").style.display="none"
  }
  if (infected!==0){
    document.getElementById("in-sum").innerText=`${infected} ${infected>1?" sections are ":"section is "} infected `;
  }

  if (a!==0){
    document.getElementById("k").style.display="flex"
    updateSection("k");
    changeDisplay("k")
  }
  else{
    hideEmptyCb("k")
    document.getElementById("k").style.display="none"
  }
  if (b!==0){
    document.getElementById("l").style.display="flex"
    // updateb()
    updateSection("l")
    changeDisplay("l")
  }
  else{
    hideEmptyCb("b")
    document.getElementById("b").style.display="none"
  }
  if (c!==0){
    document.getElementById("c").style.display="flex"
    // updatec()
    updateSection("c")
    changeDisplay("c")
  }
  else{
    hideEmptyCb("c")
    document.getElementById("c").style.display="none"
  }
  if (d!==0){
    document.getElementById("d").style.display="flex"
    // updated()
    updateSection("d")
    changeDisplay("d")  
  }
  else{
    hideEmptyCb("d")
    document.getElementById("d").style.display="none"
  }

  if (newa.length || newb.length || newc.length|| newd.length|| recover.length){
    if (recover.length>0)document.getElementById("recv").innerHTML=`${recover.length>1?`Section ${recover} have`:`Sections ${recover} has`} <span id="recg">recovered</span>`
    if (newa.length || newb.length || newc.length|| newd.length){
      document.getElementById("newi").innerHTML=
      `${newa.length + newb.length +newc.length +newd.length} ${(newa.length + newb.length +newc.length +newd.length)>1?"Sections are ":"Section is "}<span id="newr">infected</span> recently:`
    }
    if (newa.length){
      document.getElementById("newa").setAttribute("style","display:li")
      document.getElementById("newa").innerHTML=`${newa.length>1?`Section ${newa} are`:`Sections ${newa} is`} <span id="newr">infected</span> with <span id="newr-a">Bacterial Blight</span>`
    }
    else{
      document.getElementById("newa").setAttribute("style","display:none")
    }
    if(newb.length){
      document.getElementById("newb").setAttribute("style","display:li")
      document.getElementById("newb").innerHTML=`${newb.length>1?`Section ${newb} are`:`Sections ${newb} is`} <span id="newr">infected</span> with <span id="newr-b">Brown Streak Disease</span>`
    }
    else{
      document.getElementById("newb").setAttribute("style","display:none")
    }
    if(newc.length){
      document.getElementById("newc").setAttribute("style","display:li")
      document.getElementById("newc").innerHTML=`${newc.length>1?`Section ${newc} are`:`Sections ${newc} is`} <span id="newr">infected</span> with <span id="newr-c">Green Mottle</span>`
    }
    else{
      document.getElementById("newc").setAttribute("style","display:none")
    }
    if(newd.length){
      document.getElementById("newd").setAttribute("style","display:li")
      document.getElementById("newd").innerHTML=`${newd.length>1?`Section ${newd} are`:`Sections ${newd} is`} <span id="newr">infected</span> with <span id="newr-d">Mosaic Disease</span>`
    }
    else{
      document.getElementById("newd").setAttribute("style","display:none")
    }
  }
  else{
    document.getElementById("nonew").innerHTML=`There are no new changes in the sections's status.`
  }

  
  
  document.getElementById("now").innerText="Latest Record On " + datetime
  document.getElementById("total").innerText=`Out of ${sectioncount} ${sectioncount>1?"sections ":"section "}:`

}




function showPie() {
  chart_data=[{
    type: "pie",
    startAngle: 240,
    yValueFormatString: "##0.00\"%\"",
    indexLabel: "{label} {y}",
    dataPoints: [

    ]
  }]
  
  if (healthycount>0)chart_data[0].dataPoints.push({color:"#008FFB",y: `${percentageCal(healthycount)}`, label: "Healthy"})
  if (a>0)chart_data[0].dataPoints.push({color:"rgb(0, 227, 150)",y: `${percentageCal(a)}`, label: "Bacterial Blight"})
  if (b>0)chart_data[0].dataPoints.push({color:"rgb(254, 176, 25)",y: `${percentageCal(b)}`, label: "Brown Streak Disease"})
  if (c>0)chart_data[0].dataPoints.push({color:"rgb(255, 69, 96)",y: `${percentageCal(c)}`, label: "Green Mottle"})
  if (d>0)chart_data[0].dataPoints.push({y: `${percentageCal(d)}`, label: "Mosaic Disease"})
  // console.log(chart_data[0].dataPoints)
  var chart = new CanvasJS.Chart("chartContainer", {
    animationEnabled: true,
    title: {
      text: "Cassava Farm Record",
      fontColor: "green",
    },
    data: chart_data
    
  });
  
  chart.render();
}


//calculate % for pie chart
function percentageCal(num){
  return num/sectioncount*100
}


//keeping the upload img source
var imgsrc;



function readURL(input) {
    if (input.files && input.files[0]) {
  
      var reader = new FileReader();
  
      

      reader.onload = function(e) {
        imgsrc=e.target.result;
        $('.image-upload-wrap').hide();
  
        $('.file-upload-image').attr('src', imgsrc);
        $('.file-upload-content').show();
  
        $('.image-title').html(input.files[0].name);
        //document.getElementById("directthis").setAttribute("src", imgsrc);
        console.log("this the upload one:",imgsrc)
      };
      
      reader.readAsDataURL(input.files[0]);
      
      testDirect()
      

    } else {
      removeUpload();
    }
  }

function hideEmptyCb(status){
  document.getElementById(status+"cb").setAttribute("style","display:none")
}

function changeDisplay(status){
  // console.log(status)
  document.getElementById(status+"cb").setAttribute("style","display:flex")
  

  var displaySec = document.getElementById(status)
  if (displaySec.style.display!=="none")displaySec.setAttribute("style","display:none")
  else {
    if ((status==="hl" && healthycount!==0)||
    (status==="k"&&a!==0)||
    (status==="l"&&b!==0)||
    (status==="c"&&c!==0)||
    (status==="d"&&d!==0))
    displaySec.setAttribute("style","display:flex")
  }
}

function removeUpload() {
    $('.file-upload-input').replaceWith($('.file-upload-input').clone());
    $('.file-upload-content').hide();
    $('.image-upload-wrap').show();
}

function zoomSection(isrc,sec){
  // console.log("event work")
  // console.log(isrc)
  var zoomAtr = "background-image:url(\""+isrc+"\")"
  // console.log(zoomAtr)
  document.getElementById("zoomC").setAttribute("style",`${zoomAtr}`)
  // console.log(document.getElementById("zoomC"))
  if (document.getElementById("mydiv").style.display!=="block")
    document.getElementById("mydiv").setAttribute("style","display:block")
  document.getElementById("zoomtitle").innerText=`Section ${sec} zoom out`
  
}

function resetSecDisplay(){
  var idArray = ["hl2","k2","l2","c2","d2"]
  idArray.forEach(element =>{
    document.getElementById(element).innerHTML=""
  })

  document.getElementById("colchart").setAttribute("style","display:none")
}


var chart1;
function showColChart(){
  var options = {
    series: [{
    name: 'Healthy',
    data: hrec
  }, {
    name: 'Bacterial Blight',
    data: arec
  }, {
    name: 'Brown Streak Disease',
    data: brec
  },{
    name: 'Green Mottle',
    data: crec
  }, {
    name: 'Mosaic Disease',
    data: drec
  }],
    chart: {
      type: 'bar',
      height: 450,
      stacked: true,
    },
    plotOptions: {
      bar: {
        horizontal: true,
        dataLabels: {
          total: {
            enabled: true,
            offsetX: 0,
            style: {
              fontSize: '13px',
              fontWeight: 900
            }
          }
        }
      },
    },
    stroke: {
      width: 1,
      colors: ['#fff']
    },
    title: {
      // text: 'Fiction Books Sales'
    },
    xaxis: {
      categories: lastRecDate,
      labels: {
        formatter: function (val) {
          return val
        }
      }
    },
    yaxis: {
      title: {
        text: undefined
      },
    },
    tooltip: {
      y: {
        formatter: function (val) {
          return val
        }
      }
    },
    fill: {
      opacity: 1
    },
    legend: {
      position: 'top',
      horizontalAlign: 'left',
      offsetX: 40
    }
  };
 
  
  // var chart = new ApexCharts(document.querySelector("#colchart"), options);
  chart1 = new ApexCharts(document.querySelector("#colchart"), options);
  document.getElementById("colchart").setAttribute("style","display:block")
  

}

function updateChart(){
  showColChart()
  chart1.render()
  window.resizeTo(2216-10,1069-10)
}

function updateCurrent(){
  var disList =["healthy","a","b","c","d"]
  //data[0].status =disList[Math.floor(Math.random() * disList.length)]
  console.log(disList[Math.floor(Math.random() * disList.length)])
}

// reload with interval
function pageReload(){
  healthycount=0,infected=0,a=0,b=0,c=0,d=0,sectioncount=0
  hrec=[],arec=[],brec=[],crec=[],drec=[],lastRecDate=[]; 
  datetime;
  recover=[],newa=[],newb=[],newc=[],newd=[];
  hdata=[],adata=[],bdata=[],cdata=[],ddata=[];
  
  getRecord()
  resetSecDisplay()
  classifiData(data)
  showPie()
  showStatus(data)
  showColChart()
  window.resizeTo(2016,969)
  updateChart()
  
  // chart1.render();
  
  console.log(healthycount)
  console.log(window.innerHeight)
  // console.log(document.getElementById("b"))
  
}

pageReload()

// Make the DIV element draggable:
dragElement(document.getElementById("mydiv"));

function dragElement(elmnt) {
  var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  if (document.getElementById(elmnt.id + "header")) {
    // if present, the header is where you move the DIV from:
    document.getElementById(elmnt.id + "header").onmousedown = dragMouseDown;
  } else {
    // otherwise, move the DIV from anywhere inside the DIV:
    elmnt.onmousedown = dragMouseDown;
  }

  function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    // get the mouse cursor position at startup:
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag;
  }

  function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    // calculate the new cursor position:
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    // set the element's new position:
    elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
    elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
  }

  function closeDragElement() {
    // stop moving when mouse button is released:
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

function setDragDisplay(state){
  document.getElementById("mydiv").setAttribute("style",`display:${state}`)
}

//hover zoom
var addZoom = target => {
  // (A) GET CONTAINER + IMAGE SOURCE
  let container = document.getElementById(target),
      imgsrc = container.currentStyle || window.getComputedStyle(container, false);
      imgsrc = imgsrc.backgroundImage.slice(4, -1).replace(/"/g, "");
 
  // (B) LOAD IMAGE + ATTACH ZOOM
  let img = new Image();
  img.src = imgsrc;
  img.onload = () => {
    // (B1) CALCULATE ZOOM RATIO
    let ratio = img.naturalHeight / img.naturalWidth,
        percentage = ratio * 100 + "%";
 
    // (B2) ATTACH ZOOM ON MOUSE MOVE
    container.onmousemove = e => {
      let rect = e.target.getBoundingClientRect(),
          xPos = e.clientX - rect.left,
          yPos = e.clientY - rect.top,
          xPercent = xPos / (container.clientWidth / 100) + "%",
          yPercent = yPos / ((container.clientWidth * ratio) / 100) + "%";
 
      Object.assign(container.style, {
        backgroundPosition: xPercent + " " + yPercent,
        backgroundSize: img.naturalWidth + "px"
      });
    };
 
    // (B3) RESET ZOOM ON MOUSE LEAVE
    container.onmouseleave = e => {
      Object.assign(container.style, {
        backgroundPosition: "center",
        backgroundSize: "cover"
      });
    };
  }
};
 
// (C) ATTACH FOLLOW ZOOM
window.onload = () => addZoom("zoomC");




// {
// $('.image-upload-wrap').bind('dragover', function () {
//       $('.image-upload-wrap').addClass('image-dropping');
//     });
//     $('.image-upload-wrap').bind('dragleave', function () {
//       $('.image-upload-wrap').removeClass('image-dropping');
// });
// }


// setInterval(pageReload,300000)




// const upload = async () => {
async function testDirect(){
  const inputFile = document.getElementById('input-img').files[0]
  if (inputFile) {
      const formdata = new FormData();
      formdata.append('file', inputFile, inputFile.name);
      const res = await fetch('http://127.0.0.1:8000/classfication/', {
        method: 'POST',
        body: formdata,
        redirect: 'follow',
        mode: 'cors',
      });
      data = await res.json();
      console.log(data);
  }
  else console.log('Empty file input')
}

async function getRecord(){
  
  const formdata = new FormData();
  const res = await fetch('http://127.0.0.1:8000/records/', {
    method: 'GET',
  });
  data = await res.json();
  console.log(data);
}
  
