"use client"
import { useState } from 'react'

function classNames(...classes) {
  return classes.filter(Boolean).join(' ')
}

export default function Example() {
  const [loading, setLoading] = useState(false);
  const [agreed, setAgreed] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true); // Set loading state to true
    setResult(null);

    const formData = new FormData(event.target);
    const sentence1 = formData.get('sentence1');
    const sentence2 = formData.get('sentence2');

    try {
      const response = await fetch('http://127.0.0.1:8000/checkparaphrase', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sentence1, sentence2 }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }

      const data = await response.json();
      setResult(data);
      setError(null);
    } catch (error) {
      setError('An error occurred while fetching data');
      setResult(null);
    } finally {
      setLoading(false); // Set loading state to false after API response
    }
  };

  return (
    <div className="isolate bg-white px-6 py-24 sm:py-32 lg:px-8">
      <div
        className="absolute inset-x-0 top-[-10rem] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[-20rem]"
        aria-hidden="true"
      >
        <div
          className="relative left-1/2 -z-10 aspect-[1155/678] w-[36.125rem] max-w-none -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%-40rem)] sm:w-[72.1875rem]"
          style={{
            clipPath:
              'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)',
          }}
        />
      </div>
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Paraphrase Detection</h2>
        <p className="mt-2 text-lg leading-8 text-gray-600">
          Enter your sentences below and click on the submit button to see your result.
        </p>
      </div>
      <form onSubmit={handleSubmit} className="mx-auto mt-16 max-w-xl sm:mt-20">
        <div className="grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <label htmlFor="sentence1" className="block text-sm font-semibold leading-6 text-gray-900">
              Sentence 1
            </label>
            <div className="mt-2.5">
              <textarea
                name="sentence1"
                id="sentence1"
                rows={4}
                className="block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                defaultValue={''}
              />
            </div>
          </div>
          <div className="sm:col-span-2">
            <label htmlFor="sentence2" className="block text-sm font-semibold leading-6 text-gray-900">
              Sentence 2
            </label>
            <div className="mt-2.5">
              <textarea
                name="sentence2"
                id="sentence2"
                rows={4}
                className="block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                defaultValue={''}
              />
            </div>
          </div>
        </div>
        <div className="mt-10">
          <button
            type="submit"
            className="block w-full rounded-md bg-indigo-600 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          >
            {loading ? (
              <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-8 border-t-4 border-b-4 border-white"></div>
            </div>
            ) : (
              'Submit'
            )}
          </button>
        </div>
      </form>
      {result && (
        <div className="mt-8 text-center">
          <p className="text-lg font-semibold text-indigo-600">Result:</p>
          <p className="mt-2 text-gray-600">Similarity - {result.parphrase_probability}%</p>
          <p className="mt-2 text-gray-600">Label - {result.paraphrase_result}</p>
        </div>
      )}

      {error && (
        <div className="mt-8 text-center">
          <p className="text-lg font-semibold text-red-600">Error:</p>
          <p className="mt-2 text-gray-600">{error}</p>
        </div>
      )}
    </div>
  )
}
