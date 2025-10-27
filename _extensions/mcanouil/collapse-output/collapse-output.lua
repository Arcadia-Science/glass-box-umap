--[[
MIT License

Copyright (c) 2025 Mickaël Canouil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

--- @type string The method to use for collapsing output (lua or javascript)
local method = "lua"

--- Check if a string is empty or nil
--- @param s string|nil The string to check
--- @return boolean true if the string is nil or empty
local function is_empty(s)
  return s == nil or s == ''
end

--- Extract metadata value from document meta using nested structure
--- @param meta table The document metadata table
--- @param key string The metadata key to retrieve
--- @return string|nil The metadata value as a string, or nil if not found
local function get_metadata_value(meta, key)
  -- Check for the nested structure: extensions.collapse-output.key
  if meta['extensions'] and meta['extensions']['collapse-output'] and meta['extensions']['collapse-output'][key] then
    return pandoc.utils.stringify(meta['extensions']['collapse-output'][key])
  end

  return nil
end

--- Get configuration from metadata
--- This function extracts the configuration options from document metadata
--- @param meta table The document metadata table
--- @return table The metadata table (unchanged)
function get_configuration(meta)
  local meta_method = get_metadata_value(meta, 'method')

  -- Set method
  if not is_empty(meta_method) then
    method = (meta_method --[[@as string]]):lower()
    if method ~= "lua" and method ~= "javascript" then
      quarto.log.warning("Invalid method '" .. method .. "'. Using default 'lua'.")
      method = "lua"
    end
  else
    method = "lua" -- default method
  end

  return meta
end

--- Ensure HTML dependencies are included (for javascript method)
--- @return nil
local function ensure_html_deps()
  if method == "javascript" then
    quarto.doc.add_html_dependency({
      name = 'collapse-output',
      version = '1.0.0',
      scripts = {
        { path = "collapse-output.min.js", afterBody = true }
      }
    })
  end
end

--- Process Div elements to add collapse functionality
--- @param div pandoc.Div The Div element to process
--- @return pandoc.Div|nil The modified Div or nil if no changes
function process_div(div)
  if not quarto.doc.is_format("html") then
    return nil
  end

  if div.attributes["output-fold"] ~= "true" then
    return nil
  end

  local summary_text = div.attributes["output-summary"] or "Code Output"

  if method == "lua" then
    -- Process with Lua
    local new_content = {}

    for _, block in ipairs(div.content) do
      if block.t == "Div" then
        local has_cell_output = false
        has_cell_output = block.classes:find("cell-output") or
          block.classes:find("cell-output-stdout") or
          block.classes:find("cell-output-display")

        if has_cell_output then
          table.insert(new_content, pandoc.RawBlock("html", "<details><summary>" .. summary_text .. "</summary>"))
          table.insert(new_content, block)
          table.insert(new_content, pandoc.RawBlock("html", "</details>"))
        else
          table.insert(new_content, block)
        end
      else
        table.insert(new_content, block)
      end
    end

    div.content = new_content
    return div
  elseif method == "javascript" then
    ensure_html_deps()
    return div
  else
    return nil
  end
end

--- Pandoc filter configuration
--- Defines the order of filter execution:
--- 1. Get configuration from metadata
--- 2. Process Div elements for collapse functionality
return {
  { Meta = get_configuration },
  { Div = process_div }
}
