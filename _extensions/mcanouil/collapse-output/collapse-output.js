/**
* MIT License
*
* Copyright (c) 2025 Mickaël Canouil
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
document.addEventListener('DOMContentLoaded', () => {
  const cells = document.querySelectorAll('.cell[data-output-fold="true"]');
  
  cells.forEach((cell) => {
    const output =
      cell.querySelector('.cell-output-display') ||
      cell.querySelector('.cell-output') ||
      cell.querySelector('.cell-output-stdout');
    
    if (output) {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = cell.dataset.outputSummary || 'Code Output';
      
      details.appendChild(summary);
      output.parentNode.insertBefore(details, output);
      details.appendChild(output);
    }
  });
});
