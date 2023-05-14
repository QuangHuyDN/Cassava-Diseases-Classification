const upload = async () => {
    const inputFile = document.getElementById('image').files[0]
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