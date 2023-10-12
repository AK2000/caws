(async () =>  {
    var endpoint_links = document.querySelectorAll('div[class="h3 my-0"] a[href*="endpoints"]');

    Array.prototype.forEach.call(endpoint_links, async function (element, index) {

        // step 1: get endpoint id
        var segments = element.href.split('/');
        var endpoint = segments.pop() || segments.pop();

        // step 2: query server for endpoint stats
        let url = `https://129.114.27.161:5001/caws?endpoint=${endpoint}`
        const response = await fetch(url);
        const endpoint_json = await response.json();

        // step 3: display results
        var parentEnergyElement = element.parentNode.parentNode.nextElementSibling.firstElementChild;

        energyContentDiv = parentEnergyElement.querySelector('.caws-energy');
        consumedContentDiv = parentEnergyElement.querySelector('.caws-consumed');

        if (energyContentDiv != null || consumedContentDiv != null) {
            // Edit value in existing div

            if (energyContentDiv != null)
                energyContentDiv.innerHTML = endpoint_json.total_energy;

            if (consumedContentDiv != null)
                consumedContentDiv.innerHTML = endpoint_json.energy_consumed;
        }
        else {
            // Create parent div
            var energyDiv = document.createElement('div');
            energyDiv.classList.add('caws');
            energyDiv.classList.add('col-lg');
            energyDiv.classList.add('py-1');

            var energyRowDiv = document.createElement('div');
            energyRowDiv.classList.add('row');

            // title div
            var energyTitleDiv = document.createElement('div');
            energyTitleDiv.classList.add('col-4');
            energyTitleDiv.classList.add('col-lg-12');
            energyTitleDiv.classList.add('small');
            energyTitleDiv.classList.add('text-uppercase');
            energyTitleDiv.classList.add('text-success');
            energyTitleDiv.innerHTML = 'Total node energy expenditure (kWh) <img src="https://www.svgrepo.com/show/923/leaf.svg" alt="leaf" width="13px" heigh="13px" />';
            energyRowDiv.appendChild(energyTitleDiv);

            // content div
            var energyContentDiv = document.createElement('div');
            energyContentDiv.classList.add('caws-energy')
            energyContentDiv.classList.add('col-8');
            energyContentDiv.classList.add('col-lg-12');
            energyContentDiv.classList.add('small');
            energyContentDiv.innerHTML = endpoint_json.total_energy;
            energyRowDiv.appendChild(energyContentDiv);
            energyDiv.appendChild(energyRowDiv);
            parentEnergyElement.appendChild(energyDiv);

            var consumedDiv = document.createElement('div');
            consumedDiv.classList.add('caws');
            consumedDiv.classList.add('col-lg');
            consumedDiv.classList.add('py-1');

            var consumedRowDiv = document.createElement('div');
            consumedRowDiv.classList.add('row');

            var consumedTitleDiv = document.createElement('div');
            consumedTitleDiv.classList.add('col-4');
            consumedTitleDiv.classList.add('col-lg-12');
            consumedTitleDiv.classList.add('small');
            consumedTitleDiv.classList.add('text-uppercase');
            consumedTitleDiv.classList.add('text-success');
            consumedTitleDiv.innerHTML = 'Energy consumed by tasks (kWH) <img src="https://www.svgrepo.com/show/923/leaf.svg" alt="leaf" width="13px" heigh="13px" />';
            consumedRowDiv.appendChild(consumedTitleDiv);

            // content div
            var consumedContentDiv = document.createElement('div');
            consumedContentDiv.classList.add('caws-consumed')
            consumedContentDiv.classList.add('col-8');
            consumedContentDiv.classList.add('col-lg-12');
            consumedContentDiv.classList.add('small');
            consumedContentDiv.innerHTML = endpoint_json.energy_consumed;
            consumedRowDiv.appendChild(consumedContentDiv); 
            consumedDiv.appendChild(consumedRowDiv);
            parentEnergyElement.appendChild(consumedDiv);
        }

    }, endpoint_links);

})();
