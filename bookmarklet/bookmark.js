(async () =>  {
    var endpoint_links = document.querySelectorAll('div[class="h3 my-0"] a[href*="endpoints"]');

    Array.prototype.forEach.call(endpoint_links, async function (element, index) {

        // step 1: get endpoint id
        var segments = element.href.split('/');
        var endpoint = segments.pop() || segments.pop();

        // step 2: query server for endpoint stats
        let url = `https://129.114.109.109:5001/caws?endpoint=${endpoint}`
        const response = await fetch(url);
        const endpoint_json = await response.json();

        // step 3: display results
        // Create parent div
        // TODO: handle case where div already exists
        var energyDiv = document.createElement('div');
        energyDiv.classList.add('caws');
        energyDiv.classList.add('col-lg');

        var energyRowDiv = document.createElement('div');
        energyRowDiv.classList.add('row');

        // title div
        var energyTitleDiv = document.createElement('div');
        energyTitleDiv.classList.add('col-4');
        energyTitleDiv.classList.add('col-lg-12');
        energyTitleDiv.classList.add('small');
        energyTitleDiv.classList.add('text-uppercase');
        energyTitleDiv.classList.add('text-success');
        energyTitleDiv.innerHTML = 'Total energy expenditure';
        energyRowDiv.appendChild(energyTitleDiv);

        // content div
        var energyContentDiv = document.createElement('div');
        energyContentDiv.classList.add('col-8');
        energyContentDiv.classList.add('col-lg-12');
        energyContentDiv.classList.add('small');
        energyContentDiv.innerHTML = endpoint_json.total_energy;
        energyRowDiv.appendChild(energyContentDiv);

        energyDiv.appendChild(energyRowDiv);

        var elementSibling = element.parentNode.parentNode.nextElementSibling.firstElementChild.lastElementChild;
        elementSibling.parentNode.insertBefore(energyDiv, elementSibling.nextSibling);

    }, endpoint_links);

})();
