import logo from "./logo.svg";
import "./App.css";
import { useEffect, useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [data, setData] = useState(null);
  const [dataList, setDataList] = useState([]);

  const [isLoading, setIsLoading] = useState(false);

  const [toggle, setToggle] = useState(true);

  const toggleClass = "transform translate-x-6";

  const [fileName, setFileName] = useState("");
  const [fileInputs, setFileInputs] = useState([]);

  const [resultData, setResultData] = useState(null);

  const analyseSentence = async () => {
    setIsLoading(true);

    if (fileInputs.length > 0) {
      for (let sentence of fileInputs) {
        if (sentence === null || sentence.length === 0)
          continue;
          
        const response = await axios.post("/predict", {
          sentences: [sentence],
        });

        setResultData(response.data);
      }

      setFileName("");
      setFileInputs([]);
      setIsLoading(false);
    } else {
      axios
        .post("/predict", {
          sentences: [text],
        })
        .then((res) => {
          setDataList([res.data, ...dataList]);
          setIsLoading(false);
        });
    }

    setText('');
  };

  useEffect(() => {
    if (resultData !== null) {
      setDataList([resultData, ...dataList]);
    }
  }, [resultData]);

  const loadFile = (e) => {
    const file = e.target.files[0];

    setFileName(file.name);

    const reader = new FileReader();

    reader.onload = (e) => {
      const text = e.target.result;
      setFileInputs(text.split("\n"));
    };

    reader.readAsText(file);
  };

  const _handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      analyseSentence();
    }
  }

  const processDataAsComponents = () => {
    let components = [];

    for (let data of dataList) {
      console.log(dataList);
      let tokens = [];
      let currentData = "";

      let isAspect = false;
      let isBeginning = false;
      let isPositive = false;
      let isNegative = false;
      let isNeutral = false;

      for (let i = 0; i < data["tokens"].length; i++) {
        if (data["ate_outputs"][i] !== 0) {
          if (isBeginning === false) {
            isBeginning = true;
          } 
          else if (data["ate_outputs"][i] < 2) {
            continue;
          }

          currentData += data["tokens"][i] + " ";
          isAspect = true;

          switch (data["sa_outputs"][i]) {
            case 1:
              isNegative = true;
              break;
            case 2:
              isNeutral = true;
              break;
            case 3:
              isPositive = true;
              break;
            default:
              break;
          }
        } else {
          if (currentData.length === 0) {
            currentData = data["tokens"][i];
          }
          
          let classValue = "mx-1 py-2";
          
          if (isAspect) {
            classValue += " mx-1 underline underline-offset 1 font-bold";
            i--;
          }

          if (isNegative) 
            classValue += " p-2 rounded-xl bg-red-500 font-bold";
          else if (isNeutral)
            classValue += " p-2 rounded-xl bg-yellow-500 font-bold";
          else if (isPositive)
            classValue += " p-2 rounded-xl bg-green-500 font-bold";

          tokens = [
            ...tokens,
            <span key={`token-${i}`} className={classValue}>
              {currentData}
            </span>,
          ];

          classValue = '';
          currentData = '';
          isAspect = false;
          isBeginning = false;
          isPositive = false;
          isNegative = false;
          isNeutral = false;
        }
      }

      if (currentData.length > 0) {
        let classValue = "mx-1 py-2";

        if (isAspect) {
          classValue += " mx-1 underline underline-offset 1 font-bold";
        }

        if (isNegative) classValue += " p-2 rounded-xl bg-red-500 font-bold";
        else if (isNeutral)
          classValue += " p-2 rounded-xl bg-yellow-500 font-bold";
        else if (isPositive)
          classValue += " p-2 rounded-xl bg-green-500 font-bold";

        tokens = [
          ...tokens,
          <span key={`token-${data["tokens"].length - 1}`} className={classValue}>
            {currentData}
          </span>,
        ];

        classValue = '';
        currentData = '';
        isBeginning = false;
        isAspect = false;
        isPositive = false;
        isNegative = false;
        isNeutral = false;
    }

      components = [...components, tokens];
    }

    console.log(components)

    return components;
  };

  return (
    <div className="App">
      <div className="w-full align-center justify-center mt-2 text-center">
        <form className="bg-white shadow-md max-w-xl mx-auto rounded px-8 pt-6 pb-8 mb-4" onSubmit={event => event.preventDefault()} onKeyDown={_handleKeyDown}>
          <div className="mb-4">
            <label
              className="block text-gray-700 text-sm font-bold mb-2"
              for="username"
            >
              Sentence to be analysed
            </label>
            {toggle ? (
              <input
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                id="sentence"
                type="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="I liked the food"
              />
            ) : (
              <div class="flex w-full items-center justify-center bg-grey-lighter">
                <label class="w-64 flex flex-col items-center px-4 py-6 bg-blue-500 text-white rounded-lg shadow-lg tracking-wide uppercase border border-blue cursor-pointer hover:bg-blue hover:text-white">
                  <svg
                    class="w-8 h-8"
                    fill="currentColor"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                  >
                    <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
                  </svg>
                  <span class="mt-2 text-base leading-normal">
                    {fileName.length > 0 ? fileName : "Select a file"}
                  </span>
                  <input
                    type="file"
                    class="hidden"
                    onChange={(e) => loadFile(e)}
                  />
                </label>
              </div>
            )}
          </div>
          <div className="flex items-center flex-row justify-between">
            <div className="col flex">
              <div
                className={`md:w-14 md:h-7 w-12 h-6 flex items-center rounded-full p-1 cursor-pointer transition ease-in-out delay-50 " + ${
                  toggle ? "bg-gray-300" : "bg-blue-500"
                }`}
                onClick={() => {
                  setToggle(!toggle);
                }}
              >
                <div
                  className={
                    "bg-white md:w-6 md:h-6 h-5 w-5 rounded-full shadow-md transform" +
                    (toggle ? null : toggleClass)
                  }
                ></div>
              </div>
              <div className="ml-2">Load .txt file</div>
            </div>
            <div className="col">
              <button
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline flex align-middle"
                type="button"
                disabled={isLoading}
                onClick={analyseSentence}
              >
                {isLoading ? "Analysing..." : "Analyse"}
              </button>
            </div>
          </div>
        </form>
        {processDataAsComponents().map((data) => (
          <div className="bg-white shadow-md max-w-7xl mx-auto rounded px-8 pt-6 pb-8 mb-4 flex flex-wrap align-baseline">
            {
              data.map(x => x)
            }
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
